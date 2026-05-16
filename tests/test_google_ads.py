from pathlib import Path

import numpy as np

from site_audit.google_ads import GoogleAdsClient, GoogleAdsConfig, build_analysis


def test_google_ads_client_fetches_and_caches_search_terms(tmp_path: Path) -> None:
    calls = []

    def requester(url, body, headers):
        calls.append((url, body, headers))
        return {
            "results": [
                {
                    "searchTermView": {"searchTerm": "help desk software"},
                    "segments": {"keyword": {"info": {"text": "help desk software"}}},
                    "campaign": {"name": "Support"},
                    "adGroup": {"name": "Help Desk"},
                    "customer": {"currencyCode": "USD"},
                    "metrics": {
                        "costMicros": "12500000",
                        "clicks": "5",
                        "impressions": "100",
                        "conversions": "1",
                        "conversionsValue": "99",
                    },
                }
            ]
        }

    config = GoogleAdsConfig(
        developer_token="dev",
        client_id="client",
        client_secret="secret",
        refresh_token="refresh",
        customer_id="123-456-7890",
        login_customer_id="999-888-7777",
        start_date="2026-01-01",
        end_date="2026-01-31",
        refresh=True,
    )
    snapshot = GoogleAdsClient(
        tmp_path,
        requester=requester,
        token_getter=lambda creds: "access",
    ).load_or_fetch(config)

    assert snapshot["meta"]["status"] == "ok"
    assert snapshot["raw"]["search_terms"][0]["searchTermView"]["searchTerm"] == "help desk software"
    assert calls[0][0].endswith("/v22/customers/1234567890/googleAds:search")
    assert calls[0][2]["login-customer-id"] == "9998887777"
    assert "ORDER BY metrics.cost_micros DESC" in calls[0][1]["query"]


def test_google_ads_analysis_exposes_paid_terms_as_keyword_demand() -> None:
    snapshot = {
        "meta": {"status": "ok", "provider": "google_ads", "params": {"start_date": "2026-01-01", "end_date": "2026-01-31"}},
        "raw": {
            "search_terms": [
                {
                    "searchTermView": {"searchTerm": "affiliate software"},
                    "segments": {"keyword": {"info": {"text": "affiliate tracking"}}},
                    "campaign": {"name": "Affiliate"},
                    "adGroup": {"name": "Software"},
                    "customer": {"currencyCode": "EUR"},
                    "metrics": {"costMicros": "25500000", "clicks": "10", "impressions": "200", "conversions": "2", "conversionsValue": "100"},
                }
            ]
        },
    }

    analysis = build_analysis(snapshot, [], np.zeros((0, 2), dtype=np.float32))
    row = analysis.payload["organic_keywords"][0]

    assert analysis.payload["summary"]["provider"] == "google_ads"
    assert analysis.payload["summary"]["paid_cost"] == 25.5
    assert analysis.payload["summary"]["paid_conversions"] == 2
    assert analysis.payload["summary"]["paid_conversion_value"] == 100
    assert analysis.payload["summary"]["cost_per_conversion"] == 12.75
    assert row["keyword"] == "affiliate software"
    assert row["paid_cost"] == 25.5
    assert row["paid_conversions"] == 2
    assert row["paid_conversion_value"] == 100
    assert row["traffic"] == 26
    assert row["intents"] == ["commercial", "transactional"]
