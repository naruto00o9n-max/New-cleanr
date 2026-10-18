"""Validate account responses without interpreting protection pages as success.

No challenge solving, endpoint fallback, account creation or network calls live here.
Only fixed diagnostic codes are logged; provider bodies and credentials stay private.
"""
import logging
import time

LOGGER = logging.getLogger("zeus.accounts")


class QwenAccountError(RuntimeError):
    def __init__(self, code, stage, status=None):
        self.code, self.stage, self.status = code, stage, status
        messages = {
            "PROTECTION_PAGE": "أعاد مزود الخدمة صفحة تحقق حماية؛ لم يتم تأكيد العملية. يلزم إكمال التحقق بالطريقة التي يوفرها المزود.",
            "RATE_LIMITED": "الحساب المحدد بلغ حد الاستخدام. انتظر انتهاء المهلة؛ لن يتم تبديله أو إنشاء حساب بديل تلقائيًا.",
            "NO_ACCOUNT": "لا يوجد حساب محدد صالح. افتح لوحة الحسابات لإنشاء حساب بطلب صريح أو اختيار حسابك القائم.",
            "NETWORK_ERROR": "تعذر الاتصال أثناء عملية الحساب؛ لم يتم تأكيد نجاحها. راجع سجل التشخيص.",
            "HTTP_ERROR": "رفض مزود الخدمة طلب الحساب. راجع رمز التشخيص في سجل التشغيل.",
            "INVALID_RESPONSE": "أعاد مزود الخدمة استجابة غير متوقعة؛ لم يتم تأكيد العملية.",
            "PROVIDER_ERROR": "أبلغ مزود الخدمة عن فشل عملية الحساب؛ لم يتم تأكيد نجاحها.",
            "ACTIVATION_TIMEOUT": "لم تصل رسالة التفعيل خلال مهلة الانتظار؛ لن يتم إنشاء حساب آخر تلقائيًا.",
        }
        LOGGER.warning("account_flow stage=%s code=%s status=%s", stage, code, status)
        super().__init__(messages.get(code, messages["INVALID_RESPONSE"]))


def validate_account_http_response(response, stage):
    """Reject explicit protection and transport failures, without solving a challenge."""
    status = response.status_code
    if status == 429:
        raise QwenAccountError("RATE_LIMITED", stage, status)
    body = response.text[:4096].lstrip("\ufeff \t\r\n").lower()
    if any(marker in body for marker in ("aliyun_waf", "cf-chl-", "g-recaptcha", "hcaptcha", "captcha")):
        raise QwenAccountError("PROTECTION_PAGE", stage, status)
    if not 200 <= status < 300:
        raise QwenAccountError("HTTP_ERROR", stage, status)


def validate_account_response(response, stage, *, allow_list=False):
    """Require a successful JSON response; HTTP 200 alone is never sufficient."""
    validate_account_http_response(response, stage)
    status = response.status_code
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json" and not content_type.endswith("+json"):
        raise QwenAccountError("INVALID_RESPONSE", stage, status)
    try:
        payload = response.json()
    except ValueError:
        raise QwenAccountError("INVALID_RESPONSE", stage, status) from None
    if isinstance(payload, list) and allow_list:
        return payload
    if not isinstance(payload, dict):
        raise QwenAccountError("INVALID_RESPONSE", stage, status)
    inner = payload.get("data")
    inner = inner if isinstance(inner, dict) else {}
    if payload.get("code") == "RateLimited" or inner.get("code") == "RateLimited":
        raise QwenAccountError("RATE_LIMITED", stage, status)
    if payload.get("success") is False or payload.get("error") or payload.get("detail"):
        raise QwenAccountError("PROVIDER_ERROR", stage, status)
    return payload


def require_account_success(payload, stage):
    inner = payload.get("data")
    inner = inner if isinstance(inner, dict) else {}
    if payload.get("success") is not True and not (payload.get("token") or inner.get("token")):
        raise QwenAccountError("INVALID_RESPONSE", stage)
    return payload


def selected_account_token(data, service_keys, service_type, *, now=None):
    """Use precisely the selected account. Never rotate, provision or reset limits."""
    now = time.time() if now is None else now
    limit_key, active_key, _ = service_keys.get(service_type, service_keys["chat"])
    index = data.get(active_key, -1)
    accounts = data.get("accounts", [])
    if not isinstance(index, int) or not 0 <= index < len(accounts):
        raise QwenAccountError("NO_ACCOUNT", "select")
    account = accounts[index]
    if account.get(limit_key, 0) > now:
        raise QwenAccountError("RATE_LIMITED", "select")
    token = account.get("token")
    if not isinstance(token, str) or not token.strip():
        raise QwenAccountError("NO_ACCOUNT", "select")
    return token
