#!/usr/bin/env python3
"""Playwright-based Databrary session.

Launches a persistent Chromium browser, logs in, and exposes an API client
that makes all metadata calls through the browser (via page.evaluate/fetch).
This matches how Databrary's own web app authenticates — no OAuth2 credentials
or special User-Agent required.

File downloads (chunked byte-range) still use requests against the pre-signed
download URLs returned by the API.

Usage:
    from databrary_auth import DatabarySession
    with DatabarySession() as db:
        data = db.api_json("volumes/1981/folders/")
        for record in db.api_pages("volumes/1981/folders/"):
            ...
"""
import getpass
import os
import threading
from pathlib import Path

HOME_URL = "https://databrary.org/home"
API_BASE = "https://api.databrary.org"
PROFILE_DIR = Path.home() / ".databrary-browser-profile"


# ============================================================
# Login helpers
# ============================================================

def _is_logged_in(page):
    import re
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    try:
        logout = page.locator("button:visible, a:visible").filter(
            has_text=re.compile(r"log\s*out|sign\s*out", re.I)
        )
        return logout.count() > 0
    except Exception:
        return False


def _wait_briefly(page):
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass
    page.wait_for_timeout(1500)


def _login(page):
    if _is_logged_in(page):
        print("[Auth] Existing session detected.")
        return

    print("[Auth] Logging in...")

    import re
    login_btn = page.locator("button:visible, a:visible").filter(
        has_text=re.compile(r"log\s*in|sign\s*in", re.I)
    )
    if login_btn.count() > 0:
        login_btn.first.click(timeout=10000)
        _wait_briefly(page)

    page.locator('input[name="email"]:visible').wait_for(timeout=10000)

    email = os.environ.get("DATABRARY_USERNAME") or input("Databrary email: ").strip()
    password = os.environ.get("DATABRARY_PASSWORD") or getpass.getpass("Databrary password: ")

    page.locator('input[name="email"]:visible').fill(email)
    page.locator('input[name="password"]:visible').fill(password)
    del password

    submit = page.locator('button[type="submit"]:visible')
    if submit.count() > 0:
        submit.first.click(timeout=10000)
    else:
        page.locator('input[name="password"]:visible').press("Enter")

    code_input = page.locator('input[name="code"]')
    try:
        code_input.wait_for(state="visible", timeout=8000)
        code = getpass.getpass("Databrary verification code: ")
        code_input.fill(code)
        del code

        # Find the submit button belonging to the verification form
        form = code_input.locator("xpath=ancestor::form[1]")

        if form.count() > 0:
            submit = form.locator('button[type="submit"]')
            if submit.count() > 0:
                submit.first.click(timeout=10000)
            else:
                code_input.press("Enter")
        else:
            code_input.press("Enter")

        page.wait_for_timeout(3000)
    except Exception as e:
        print(f"[DEBUG] 2FA processing exception: {type(e).__name__}: {e}")

    if not _is_logged_in(page):
        print("\n[DEBUG] Authentication diagnostics")

        print("[DEBUG] URL:", page.url)

        print("[DEBUG] Alerts:")
        print(page.locator(
            '[role="alert"], .alert, .error, [aria-live]'
        ).all_inner_texts())

        print("[DEBUG] Visible dialogs:")
        print(page.locator(
            '[role="dialog"]:visible'
        ).all_inner_texts())

        print("[DEBUG] Submit buttons:")
        buttons = page.locator('button[type="submit"]:visible')
        for i in range(buttons.count()):
            print(i, repr(buttons.nth(i).inner_text()))

        raise RuntimeError("Databrary authentication failed")
    print("[Auth] Login successful.")


def _install_token_capture(context):
    captured = {"token": None}

    def on_request(request):
        try:
            if not request.url.startswith("https://api.databrary.org/"):
                return
            auth = request.all_headers().get("authorization", "")
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
                if token:
                    captured["token"] = token
        except Exception:
            pass

    context.on("request", on_request)
    return captured, on_request


# ============================================================
# DatabraySession
# ============================================================

class DatabarySession:
    """Open browser session with Databrary. Use as a context manager."""

    def __init__(self):
        self._lock = threading.Lock()
        self._local = threading.local()
        self._playwright, self._context, self._page, self._token = self._start()

    def _start(self):
        from playwright.sync_api import sync_playwright

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        PROFILE_DIR.chmod(0o700)

        pw = sync_playwright().start()
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=True,
            accept_downloads=True,
            viewport={"width": 1440, "height": 900},
        )
        captured, listener = _install_token_capture(context)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(HOME_URL, wait_until="domcontentloaded", timeout=30000)
            _wait_briefly(page)
            _login(page)

            # Navigate to a volume page to trigger authenticated API requests
            # so the token capture listener fires
            if not captured["token"]:
                page.goto("https://databrary.org/volume/1981",
                          wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(4000)

            token = captured["token"]
            if not token:
                token = getpass.getpass("Paste Databrary Bearer token: ").strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            if not token:
                raise RuntimeError("Could not obtain Databrary Bearer token")
        finally:
            context.remove_listener("request", listener)

        return pw, context, page, token

    # ----------------------------------------------------------
    # API (all calls go through the browser)
    # ----------------------------------------------------------

    def api_json(self, path):
        """Fetch a single API endpoint through the browser. Thread-safe."""
        if not path.startswith("http"):
            if not path.startswith("/"):
                path = "/" + path
            url = API_BASE + path
        else:
            url = path

        with self._lock:
            result = self._page.evaluate(
                """async ({url, token}) => {
                    try {
                        const r = await fetch(url, {
                            method: "GET",
                            credentials: "include",
                            headers: {
                                Authorization: "Bearer " + token,
                                Accept: "application/json"
                            }
                        });
                        if (!r.ok) return {ok: false, status: r.status};
                        return {ok: true, status: r.status, data: await r.json()};
                    } catch(e) {
                        return {ok: false, status: null, error: String(e)};
                    }
                }""",
                {"url": url, "token": self._token},
            )

        if not result.get("ok"):
            raise RuntimeError(
                f"Databrary API {result.get('status')} at {path}"
            )
        return result["data"]

    def fetch_text(self, url):
        """Fetch text content through the browser (for URLs that require session cookies)."""
        if url.startswith("http://"):
            url = url.replace("http://", "https://", 1)
        with self._lock:
            result = self._page.evaluate(
                """async ({url, token}) => {
                    try {
                        const r = await fetch(url, {
                            method: "GET",
                            credentials: "include",
                            headers: {Authorization: "Bearer " + token}
                        });
                        if (!r.ok) return {ok: false, status: r.status};
                        return {ok: true, text: await r.text()};
                    } catch(e) {
                        return {ok: false, error: String(e)};
                    }
                }""",
                {"url": url, "token": self._token},
            )
        if not result.get("ok"):
            raise RuntimeError(f"fetch_text failed ({result.get('status')}) at {url}")
        return result["text"]

    def api_pages(self, path):
        """Iterate all pages of a paginated API endpoint."""
        url = path

        while url:
            if url.startswith("http://api.databrary.org/"):
                url = url.replace(
                    "http://api.databrary.org/",
                    "https://api.databrary.org/",
                    1
                )

            data = self.api_json(url)
            yield from data["results"]
            url = data.get("next")

    # ----------------------------------------------------------
    # requests.Session for file downloads (pre-signed URLs)
    # ----------------------------------------------------------

    def download_session(self):
        """Return a thread-local requests.Session for file downloads."""
        import requests
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s


    def browser_download(self, trigger, destination):
        import time

        destination = Path(destination)

        if destination.exists():
            raise FileExistsError(destination)

        t0 = time.perf_counter()

        with self._page.expect_download(timeout=120000) as download_info:
            trigger(self._page)

        download = download_info.value
        t1 = time.perf_counter()

        failure = download.failure()
        if failure:
            raise RuntimeError(f"Browser download failed: {failure}")

        t2 = time.perf_counter()

        destination.parent.mkdir(parents=True, exist_ok=True)
        download.save_as(str(destination))

        t3 = time.perf_counter()

        print(f"[Timing] Start -> download event: {t1-t0:.2f}s")
        print(f"[Timing] Download event -> completion: {t2-t1:.2f}s")
        print(f"[Timing] save_as: {t3-t2:.2f}s")
        print(f"[Timing] Total: {t3-t0:.2f}s")

        return destination



    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def close(self):
        try:
            self._context.close()
        except Exception:
            pass
        try:
            self._playwright.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
