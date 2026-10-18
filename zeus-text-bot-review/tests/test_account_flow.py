"""Offline regressions: no Discord, MongoDB, signup or live provider requests."""
import ast
import asyncio
import copy
import json
import logging
from pathlib import Path
import re
import secrets
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
import uuid
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "text_bot"))
from utils.qwen_auth import (
    QwenAccountError, require_account_success, selected_account_token,
    validate_account_http_response, validate_account_response,
)

SOURCE = ROOT / "text_bot" / "bot.py"
TREE = ast.parse(SOURCE.read_text())
KEYS = {"ocr": ("ocr_limit_until", "active_ocr_index", "ocr_count"),
        "chat": ("image_limit_until", "active_image_index", "image_count")}


class Response:
    def __init__(self, payload=None, *, status=200, content_type="application/json", text=None):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.text = json.dumps(payload) if text is None else text

    def json(self):
        return json.loads(self.text)


class TransportError(Exception):
    pass


def functions(*names, **overrides):
    """Execute real top-level bot functions with isolated transport/storage adapters."""
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError("Missing function in source")
    namespace = dict(QwenAccountError=QwenAccountError,
                     validate_account_response=validate_account_response,
                     validate_account_http_response=validate_account_http_response,
                     require_account_success=require_account_success,
                     selected_account_token=selected_account_token,
                     SERVICE_KEYS=KEYS, BASE_QWEN_URL="https://provider.invalid/api/v2",
                     TEMP_API_URL="https://mail.invalid/api/v3", secrets=secrets,
                     requests=SimpleNamespace(post=Mock(), get=Mock(), RequestException=TransportError),
                     get_base_headers=lambda **kw: {}, generate_user_agent=lambda: "test",
                     time=time, uuid=uuid, re=re, json=json,
                     datetime=datetime, timezone=timezone, timedelta=timedelta)
    namespace.update(overrides)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class ResponseTests(unittest.TestCase):
    def assert_code(self, code, fn, *args, **kwargs):
        with self.assertRaises(QwenAccountError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(code, ctx.exception.code)

    def test_200_protection_page_is_not_signup_success(self):
        response = Response(text='<meta name="aliyun_waf_aa" content="private">', content_type="text/html")
        ns = functions("signup_qwen")
        ns["requests"].post.return_value = response
        self.assert_code("PROTECTION_PAGE", ns["signup_qwen"], "x@example.test", "name", "private-password")
        self.assertEqual(1, ns["requests"].post.call_count)

    def test_403_protection_has_specific_diagnosis(self):
        self.assert_code("PROTECTION_PAGE", validate_account_response,
                         Response(text="<html>captcha</html>", content_type="text/html", status=403), "signup")

    def test_html_application_shell_is_not_claimed_to_be_protection(self):
        self.assert_code("INVALID_RESPONSE", validate_account_response,
                         Response(text="<!doctype html><div id='root'></div>", content_type="text/html"), "signup")

    def test_success_status_with_provider_error_is_rejected(self):
        self.assert_code("PROVIDER_ERROR", validate_account_response,
                         Response({"success": False, "data": {"code": "not found"}}), "signup")

    def test_empty_success_payload_is_rejected(self):
        self.assert_code("INVALID_RESPONSE", require_account_success, {}, "signup")

    def test_malformed_json_is_rejected(self):
        self.assert_code("INVALID_RESPONSE", validate_account_response, Response(text="{broken"), "signup")

    def test_http_rate_limit_is_not_retried(self):
        self.assert_code("RATE_LIMITED", validate_account_response, Response({}, status=429), "signup")

    def test_json_rate_limit_is_not_success(self):
        self.assert_code("RATE_LIMITED", validate_account_response,
                         Response({"data": {"code": "RateLimited"}}), "signup")

    def test_http_endpoint_failure_is_reported(self):
        self.assert_code("HTTP_ERROR", validate_account_response, Response({}, status=404), "signup")

    def test_json_media_type_with_charset_is_supported(self):
        self.assertEqual({"success": True}, validate_account_response(
            Response({"success": True}, content_type="Application/JSON; charset=utf-8"), "signup"))

    def test_json_suffix_media_type_is_supported(self):
        self.assertEqual({"success": True}, validate_account_response(
            Response({"success": True}, content_type="application/vendor+json"), "signup"))

    def test_null_data_does_not_crash_rate_detection(self):
        ns = functions("is_rate_limited_response")
        self.assertFalse(ns["is_rate_limited_response"]({"data": None}))

    def test_top_level_and_nested_signin_tokens_are_supported(self):
        for payload in ({"token": "test-token"}, {"success": True, "data": {"token": "test-token"}}):
            ns = functions("signin_qwen")
            ns["requests"].post.return_value = Response(payload)
            self.assertEqual("test-token", ns["signin_qwen"]("x@example.test", "password"))

    def test_missing_signin_token_is_rejected(self):
        ns = functions("signin_qwen")
        ns["requests"].post.return_value = Response({"success": True, "data": None})
        self.assert_code("INVALID_RESPONSE", ns["signin_qwen"], "x@example.test", "password")

    def test_logs_do_not_include_provider_body_or_password(self):
        ns = functions("signup_qwen")
        ns["requests"].post.return_value = Response(text='<html>aliyun_waf private-token</html>', content_type="text/html")
        with self.assertLogs("zeus.accounts", level=logging.WARNING) as captured:
            self.assert_code("PROTECTION_PAGE", ns["signup_qwen"], "private-email@example.test", "name", "private-password")
        message = "\n".join(captured.output)
        self.assertIn("stage=signup", message)
        for secret in ("private-token", "private-email", "private-password"):
            self.assertNotIn(secret, message)

    def test_transport_exception_details_are_not_exposed(self):
        ns = functions("signup_qwen")
        ns["requests"].post.side_effect = TransportError("private-email@example.test")
        with self.assertRaises(QwenAccountError) as ctx:
            ns["signup_qwen"]("x@example.test", "name", "password")
        self.assertEqual("NETWORK_ERROR", ctx.exception.code)
        self.assertNotIn("private-email", str(ctx.exception))


class AccountSelectionTests(unittest.TestCase):
    def data(self):
        return {"active_ocr_index": 0, "accounts": [
            {"token": "selected", "ocr_limit_until": 200},
            {"token": "alternative", "ocr_limit_until": 0}]}

    def test_limited_selected_account_never_switches_to_available_alternative(self):
        data = self.data()
        before = copy.deepcopy(data)
        with self.assertRaises(QwenAccountError) as ctx:
            selected_account_token(data, KEYS, "ocr", now=100)
        self.assertEqual("RATE_LIMITED", ctx.exception.code)
        self.assertEqual(before, data)

    def test_waiting_until_deadline_restores_same_account(self):
        self.assertEqual("selected", selected_account_token(self.data(), KEYS, "ocr", now=200))

    def test_no_selection_never_picks_another_account(self):
        data = self.data()
        data["active_ocr_index"] = -1
        with self.assertRaises(QwenAccountError) as ctx:
            selected_account_token(data, KEYS, "ocr", now=300)
        self.assertEqual("NO_ACCOUNT", ctx.exception.code)

    def test_no_accounts_never_provisions_automatically(self):
        provision = Mock()
        ns = functions("get_valid_qwen_token", load_accounts_data=lambda _: {"accounts": []},
                       create_and_save_new_account=provision)
        with self.assertRaises(QwenAccountError):
            ns["get_valid_qwen_token"](123, "ocr")
        provision.assert_not_called()

    def test_marking_limit_preserves_selected_index_and_other_accounts(self):
        data, save = self.data(), Mock()
        ns = functions("mark_account_rate_limited", load_accounts_data=lambda _: data, save_accounts_data=save)
        ns["mark_account_rate_limited"](123, "ocr")
        self.assertEqual(0, data["active_ocr_index"])
        self.assertEqual(0, data["accounts"][1]["ocr_limit_until"])
        self.assertGreater(data["accounts"][0]["ocr_limit_until"], time.time())
        save.assert_called_once()

    def test_manual_create_is_blocked_during_existing_limit(self):
        data = self.data()
        data["accounts"][0]["ocr_limit_until"] = time.time() + 600
        mail = Mock()
        ns = functions("create_and_save_new_account", load_accounts_data=lambda _: data, create_temp_email=mail)
        with self.assertRaises(QwenAccountError) as ctx:
            ns["create_and_save_new_account"](123)
        self.assertEqual("RATE_LIMITED", ctx.exception.code)
        mail.assert_not_called()


class ProvisioningTests(unittest.TestCase):
    def flow(self):
        data = {"accounts": [], "active_ocr_index": -1}
        ns = functions("create_and_save_new_account", load_accounts_data=Mock(return_value=data),
                       save_accounts_data=Mock(), create_temp_email=Mock(return_value="x@example.test"),
                       signup_qwen=Mock(return_value=True), get_activation_link=Mock(return_value="https://chat.qwen.ai/api/v1/auths/activate?code=test"),
                       activate_account=Mock(return_value=None), signin_qwen=Mock(return_value="test-token"))
        return ns, data

    def test_protection_stops_before_poll_activation_signin_or_save(self):
        ns, data = self.flow()
        ns["signup_qwen"].side_effect = QwenAccountError("PROTECTION_PAGE", "signup", 200)
        with self.assertRaises(QwenAccountError):
            ns["create_and_save_new_account"](123)
        ns["create_temp_email"].assert_called_once()
        ns["signup_qwen"].assert_called_once()
        for name in ("get_activation_link", "activate_account", "signin_qwen", "save_accounts_data"):
            ns[name].assert_not_called()
        self.assertEqual([], data["accounts"])

    def test_network_failure_never_creates_a_second_mailbox(self):
        ns, _ = self.flow()
        ns["signup_qwen"].side_effect = QwenAccountError("NETWORK_ERROR", "signup")
        with self.assertRaises(QwenAccountError):
            ns["create_and_save_new_account"](123)
        ns["create_temp_email"].assert_called_once()

    def test_account_is_saved_only_after_valid_signin(self):
        ns, data = self.flow()
        self.assertEqual("test-token", ns["create_and_save_new_account"](123))
        ns["save_accounts_data"].assert_called_once()
        self.assertEqual(1, len(data["accounts"]))
        self.assertEqual(0, data["active_ocr_index"])

    def test_signin_failure_does_not_save_account(self):
        ns, _ = self.flow()
        ns["signin_qwen"].side_effect = QwenAccountError("INVALID_RESPONSE", "signin")
        with self.assertRaises(QwenAccountError):
            ns["create_and_save_new_account"](123)
        ns["save_accounts_data"].assert_not_called()

    def test_escaped_activation_link_is_decoded(self):
        ns = functions("get_activation_link")
        ns["requests"].get.return_value = Response([{"body_html": '<a href="https://chat.qwen.ai/api/v1/auths/activate?token=test&amp;email=x">Activate</a>'}])
        self.assertEqual("https://chat.qwen.ai/api/v1/auths/activate?token=test&email=x",
                         ns["get_activation_link"]("x@example.test", max_attempts=1))

    def test_mail_failure_is_not_reported_as_missing_activation(self):
        ns = functions("get_activation_link")
        ns["requests"].get.return_value = Response({}, status=503)
        with self.assertRaises(QwenAccountError) as ctx:
            ns["get_activation_link"]("x@example.test", max_attempts=1)
        self.assertEqual("HTTP_ERROR", ctx.exception.code)

    def test_activation_timeout_is_explicit(self):
        ns = functions("get_activation_link")
        ns["requests"].get.return_value = Response([])
        with self.assertRaises(QwenAccountError) as ctx:
            ns["get_activation_link"]("x@example.test", max_attempts=1)
        self.assertEqual("ACTIVATION_TIMEOUT", ctx.exception.code)


class InterfaceTests(unittest.TestCase):
    def method(self, class_name, method_name, **overrides):
        cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == method_name))
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        namespace = dict(time=time, OWNER_ID=123, themed_embed=lambda *a, **kw: "embed",
                         AccountsView=lambda _: "view", load_accounts_data=Mock(return_value={"accounts": []}),
                         save_accounts_data=Mock(), create_and_save_new_account=Mock())
        namespace.update(overrides)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
        return namespace

    def interaction(self):
        return SimpleNamespace(user=SimpleNamespace(id=123), response=SimpleNamespace(send_message=AsyncMock()))

    def test_opening_admin_panel_does_not_provision(self):
        ns = self.method("AdminPanelView", "ocr_accounts")
        interaction = self.interaction()
        asyncio.run(ns["ocr_accounts"](SimpleNamespace(), interaction, None))
        ns["create_and_save_new_account"].assert_not_called()
        interaction.response.send_message.assert_awaited_once()

    def test_limit_button_cannot_clear_existing_cooldown(self):
        data = {"accounts": [{"ocr_limit_until": time.time() + 600}]}
        before = copy.deepcopy(data)
        ns = self.method("AccountDetailView", "unban", load_accounts_data=Mock(return_value=data))
        asyncio.run(ns["unban"](SimpleNamespace(user_id=123, acc_idx=0), self.interaction(), None))
        self.assertEqual(before, data)
        ns["save_accounts_data"].assert_not_called()

    def test_selecting_alternative_during_limit_is_not_saved(self):
        data = {"active_ocr_index": 0, "accounts": [{"ocr_limit_until": time.time() + 600}, {"ocr_limit_until": 0}]}
        ns = self.method("AccountDetailView", "set_active", load_accounts_data=Mock(return_value=data))
        asyncio.run(ns["set_active"](SimpleNamespace(user_id=123, acc_idx=1), self.interaction(), None))
        ns["save_accounts_data"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
