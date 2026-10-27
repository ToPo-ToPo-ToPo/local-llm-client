#!/usr/bin/env python3
"""実環境の urllib と LLMClient の既定 HTTP 層で eMudhra 14 接続先を検査する。

正常対照を先に確認し、認証情報を送らず、リダイレクトは追わない。
公開資料は証明書として読むだけで、掲載コードは実行しない。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import io
import json
import pathlib
import re
import ssl
import sys
import urllib.error
import urllib.request
import zipfile

from local_llm_client import LLMClient

CONTROL = "julialang.org"
CHAIN_URL = "https://emudhra.we-have-a-problem.com/chains.zip"
PREFIXES = [f"{kind}-{generation}{suffix}" for kind in ("dv", "ov", "ev")
            for generation in ("g3b", "g3c") for suffix in ("", "-aaa")]
PREFIXES += ["dv-g3b-noct", "dv-g3b-aaa-noct"]
HOSTS = [f"{prefix}-emudhra.we-have-a-problem.com" for prefix in PREFIXES]
ROUTES = ("urllib-default", "llmclient-sdk-default")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pem_digests(text):
    return [digest(ssl.PEM_cert_to_DER_cert(pem)) for pem in re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", text, re.S)]


def peer_identity(sock):
    result = {"leaf_sha256": digest(sock.getpeercert(True))}
    getter = getattr(getattr(sock, "_sslobj", sock), "get_unverified_chain", None)
    if getter:
        # Fully verified handshake has already completed. This only reads the served chain.
        fingerprints = []
        for certificate in getter():
            encoded = certificate.public_bytes()
            fingerprints.extend(pem_digests(encoded) if isinstance(encoded, str)
                                else [digest(encoded)])
        result["served_chain_sha256"] = fingerprints
    return result


def error_details(exc):
    errors, seen, pending = [], set(), [exc]
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        errors.append({"type": f"{type(error).__module__}.{type(error).__name__}",
                       "message": str(error), "verify_code": getattr(error, "verify_code", None),
                       "verify_message": getattr(error, "verify_message", None)})
        for child in (getattr(error, "reason", None), error.__cause__, error.__context__):
            if isinstance(child, BaseException):
                pending.append(child)
    return errors


def probe(route, host, expected):
    result = {"route": route, "host": host, "url": f"https://{host}/"}
    try:
        if route == "urllib-default":
            opener = urllib.request.build_opener(NoRedirect)
            try:
                response = opener.open(result["url"], timeout=12)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                result["http_status"] = response.status
                result.update(peer_identity(response.fp.raw._sock))
        else:
            # Construct the repository's actual client; call its transport without SDK auth.
            llm = LLMClient(model="tls-audit", base_url=f"https://{CONTROL}/v1",
                            api_key="not-needed", session=False)
            try:
                transport = llm.openai._client
                request = transport.build_request("GET", result["url"], timeout=12)
                assert not any(name.lower() in ("authorization", "proxy-authorization", "cookie")
                               for name in request.headers)
                response = transport.send(request, stream=True, follow_redirects=False)
                try:
                    result["http_status"] = response.status_code
                    sock = response.extensions["network_stream"].get_extra_info("ssl_object")
                    result.update(peer_identity(sock))
                finally:
                    response.close()
            finally:
                llm.close()
        result["tls"] = "connected"
        if host == CONTROL:
            result["judgment"] = "control-connected" if result["http_status"] == 200 else "unconfirmed"
        else:
            fingerprints = expected[host]
            result["expected_chain_sha256"] = fingerprints
            result["leaf_matches_published"] = result["leaf_sha256"] == fingerprints[0]
            result["intermediate_matches_published"] = fingerprints[1] in result.get("served_chain_sha256", [])
            result["judgment"] = ("accepted-confirmed" if result["leaf_matches_published"]
                                  and result["intermediate_matches_published"] else "unconfirmed")
    except Exception as exc:
        errors = error_details(exc)
        result.update(tls="failed", errors=errors)
        code = next((e["verify_code"] for e in errors if e["verify_code"] is not None), None)
        messages = " ".join(e["message"].lower() for e in errors)
        # -67820 = macOS Security.framework errSecCertificateRevoked (SecBase.h).
        if code in (2, 18, 19, 20, 21, 23, 27, 28, -67820) or ("compromis" in messages and "certif" in messages):
            result["judgment"] = "rejected-confirmed"
        elif code is not None:
            # Expiry, hostname mismatch and other certificate errors are not CA distrust.
            result["judgment"] = "other-certificate-rejection"
        else:
            result["judgment"] = "unconfirmed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--chains", type=pathlib.Path, help="公表元の chains.zip を再利用する")
    args = parser.parse_args()
    controls = [probe(route, CONTROL, {}) for route in ROUTES]
    control_ok = {r["route"]: r["judgment"] == "control-connected" for r in controls}
    data = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "executable": sys.executable, "python": sys.version, "openssl": ssl.OPENSSL_VERSION,
            "default_ca": ssl.get_default_verify_paths()._asdict(),
            "conditions": {"tls_verification": True, "redirects": False, "credentials": False},
            "packages": {}, "chain_source": CHAIN_URL, "controls": controls, "results": []}
    for name in ("local-llm-client", "openai", "httpx", "httpcore", "httpx2", "httpcore2", "truststore", "certifi"):
        try:
            data["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        if not all(control_ok.values()):
            raise RuntimeError("正常対照に未確認の経路があるため対象検査を開始しない")
        if args.chains:
            archive_bytes = args.chains.read_bytes()
        else:
            with urllib.request.build_opener(NoRedirect).open(CHAIN_URL, timeout=20) as response:
                archive_bytes = response.read(10 * 1024 * 1024)
        data["chains_zip_sha256"] = digest(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            expected = {host: pem_digests(archive.read(f"chains/{host}.pem").decode()) for host in HOSTS}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            data["results"] = list(pool.map(lambda pair: probe(*pair, expected),
                                            [(route, host) for route in ROUTES for host in HOSTS]))
    except Exception as exc:
        data["inspection_error"] = error_details(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    for row in controls + data["results"]:
        reasons = [e["verify_message"] or e["message"] for e in row.get("errors", [])]
        print(row["route"], row["host"], row["judgment"], "; ".join(reasons))
    print("log:", args.output.resolve())
    return 0 if len(data["results"]) == 28 and all(
        row["judgment"] == "rejected-confirmed" for row in data["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
