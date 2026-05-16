# Google Ads Keyword Source

Use Google Ads search terms when you want competitive analysis to focus on keywords with proven business value.

Organic tools can surface high-traffic keywords that are irrelevant to the product. Paid search spend is a stronger signal: if a company is paying for a search term, the term is usually closer to product, service, use-case, or conversion intent.

This source is optional. The audit does not touch Google Ads unless you explicitly select it.

## What It Does

When enabled, the audit:

1. Connects to the user's Google Ads account through the Google Ads API.
2. Reads Search Term View rows for the selected customer account.
3. Sorts search terms by `metrics.cost_micros`.
4. Imports clicks, impressions, cost, conversions/events, and conversion value when Google Ads exposes those metrics.
5. Converts those paid search terms into the same keyword-demand format used by Ahrefs, DataForSEO, and GSC.
6. Lets `--competitive-auto` select SERP-gap keywords from high-spend paid terms.

This answers:

> Which keywords are important enough to the business that we already pay Google for them?

## When To Use It

Use Google Ads keyword sourcing when:

- the site has active Google Search campaigns
- you want product/service relevance over raw organic traffic
- Ahrefs/DataForSEO/GSC include too many informational or irrelevant keywords
- you are deciding which SERP paragraph gaps deserve content budget

Do not use it when:

- the Ads account is not connected to the product you are auditing
- spend is dominated by branded terms only
- campaigns target a different country/language than the SEO report
- the user cannot grant API access or OAuth consent

## Commands

Use Google Ads as the explicit search provider:

```bash
site-audit run example.com \
  --search-provider google_ads \
  --google-ads-customer-id 123-456-7890 \
  --competitive-auto \
  --competitive-auto-product-seed "help desk software" \
  --competitive-auto-clusters 5
```

Allow Google Ads as an opt-in fallback in `auto` mode:

```bash
site-audit run example.com \
  --search-provider auto \
  --use-google-ads-keywords \
  --google-ads-customer-id 123-456-7890 \
  --competitive-auto
```

Limit the spend source:

```bash
site-audit run example.com \
  --search-provider google_ads \
  --google-ads-customer-id 123-456-7890 \
  --google-ads-start-date 2026-02-01 \
  --google-ads-end-date 2026-04-30 \
  --google-ads-min-cost 50 \
  --google-ads-search-terms-limit 500
```

If the account is under a manager account, also set:

```bash
--google-ads-login-customer-id 999-888-7777
```

Compare paid terms with GSC, Ahrefs, and DataForSEO in one semantic map:

```bash
site-audit run example.com \
  --search-provider all \
  --google-ads-customer-id 123-456-7890
```

In this mode, the report plots keywords from each source against page
vectors, titles, H1-H4 headings, paragraphs, and link titles. Use the
semantic-map filters to isolate only GSC, Google Ads, Ahrefs, DataForSEO,
or site-content entities. When Google Ads data includes conversions, the
semantic map can size or color paid keywords by traffic/volume, spend,
events, sales value, clicks, or impressions.

## Environment Variables

Store credentials in `.env` or the shell:

```bash
GOOGLE_ADS_DEVELOPER_TOKEN="..."
GOOGLE_ADS_CLIENT_ID="...apps.googleusercontent.com"
GOOGLE_ADS_CLIENT_SECRET="..."
GOOGLE_ADS_REFRESH_TOKEN="..."
GOOGLE_ADS_CUSTOMER_ID="123-456-7890"
GOOGLE_ADS_LOGIN_CUSTOMER_ID="999-888-7777" # optional manager account
```

Optional defaults:

```bash
GOOGLE_ADS_START_DATE="2026-02-01"
GOOGLE_ADS_END_DATE="2026-04-30"
GOOGLE_ADS_API_VERSION="v22"
```

## Create The Google Ads API Access

The user must create access from their own Google account or manager account.

### 1. Confirm Account Access

Use a Google account that can access the Google Ads customer account.

If the customer is managed through an MCC/manager account, keep both IDs:

- `GOOGLE_ADS_CUSTOMER_ID`: the client account being audited
- `GOOGLE_ADS_LOGIN_CUSTOMER_ID`: the manager account used for API access

Remove dashes if you prefer. The audit accepts both `123-456-7890` and `1234567890`.

### 2. Get A Developer Token

1. Open Google Ads.
2. Go to the API Center.
3. Apply for or copy the developer token.
4. Put it into `.env` as `GOOGLE_ADS_DEVELOPER_TOKEN`.

Google requires a developer token for Google Ads API calls. New tokens may start with limited access, so the account must be approved for the level of API access needed for production accounts.

### 3. Create A Google Cloud Project

1. Open Google Cloud Console.
2. Create or select a project for this audit integration.
3. Enable the Google Ads API for the project.
4. Configure the OAuth consent screen.

For internal use, add the Google account that will authorize Ads access as a test user if the OAuth app is still in testing.

### 4. Create OAuth Client Credentials

For a local CLI workflow, create an OAuth client of type **Desktop app**.

Save:

- OAuth client ID as `GOOGLE_ADS_CLIENT_ID`
- OAuth client secret as `GOOGLE_ADS_CLIENT_SECRET`

Desktop app credentials are the simplest path for generating a refresh token for one user. A web application OAuth client can also work, but then the redirect URI and refresh-token flow need to match that web app setup.

### 5. Generate A Refresh Token

Generate an OAuth refresh token for the same Google user that has Ads access.

The required OAuth scope is:

```text
https://www.googleapis.com/auth/adwords
```

The refresh token goes into:

```bash
GOOGLE_ADS_REFRESH_TOKEN="..."
```

Keep this value private. It allows the audit to request new access tokens without asking the user to log in each time.

### 6. Test With A Small Run

Start with a low row limit:

```bash
site-audit run example.com \
  --search-provider google_ads \
  --google-ads-customer-id 123-456-7890 \
  --google-ads-search-terms-limit 25 \
  --no-snapshot
```

Check the report's Organic search demand overlay. The provider should show `Google Ads`, and keyword rows should correspond to paid search terms.

## How Keyword Selection Works

For `--competitive-auto`, Google Ads rows enter the same selection pipeline as organic keywords, with one extra business signal:

- `paid_cost` is treated as demand when ranking opportunities
- paid search terms are marked commercial and transactional
- product seeds still matter
- irrelevant terms can still be rejected by `--competitive-auto-min-relevance`

Recommended configuration:

```bash
site-audit run example.com \
  --search-provider google_ads \
  --google-ads-customer-id 123-456-7890 \
  --competitive-auto \
  --competitive-auto-product-seed "your product category" \
  --competitive-auto-product-seed "your main use case" \
  --competitive-auto-min-relevance 0.4 \
  --competitive-auto-clusters 3 \
  --competitive-auto-keywords-per-cluster 1 \
  --competitive-auto-results-per-keyword 5
```

## What The User Should Do With The Output

1. Open **SERP competitors**.
2. Check the auto-selected keyword table.
3. Reject keywords where paid spend is unrelated to SEO strategy, for example branded support queries, login queries, partner names, or campaign experiments.
4. For accepted keywords, inspect missing and partial SERP topics.
5. Rewrite the target page only when the missing topic helps the product/service page satisfy the same intent.
6. Rerun with stricter product seeds if irrelevant paid terms were selected.

## Troubleshooting

`missing_credentials`

The `.env` file is missing one or more of:

- `GOOGLE_ADS_DEVELOPER_TOKEN`
- `GOOGLE_ADS_CLIENT_ID`
- `GOOGLE_ADS_CLIENT_SECRET`
- `GOOGLE_ADS_REFRESH_TOKEN`
- `GOOGLE_ADS_CUSTOMER_ID`

`401` or OAuth refresh error

The refresh token is invalid, revoked, from the wrong OAuth client, or missing the Ads scope.

`403` developer token or permission error

The developer token may not have access, the Google user may not have Ads account access, or the manager/client account relationship is wrong.

`login-customer-id` errors

Use the manager account ID as `GOOGLE_ADS_LOGIN_CUSTOMER_ID`, not the client account ID.

No useful search terms

Check date range, campaign status, campaign type, country/language targeting, and minimum cost. Search term visibility may also be limited by Google privacy thresholds.

## Official Google References

- Google Ads API OAuth overview: https://developers.google.com/google-ads/api/docs/oauth/overview
- Single-user OAuth workflow: https://developers.google.com/google-ads/api/docs/oauth/client-library
- OAuth Playground refresh-token workflow: https://developers.google.com/google-ads/api/docs/oauth/playground
- Google Ads API developer token help: https://support.google.com/google-ads/answer/2375503
- SearchTermView fields: https://developers.google.com/google-ads/api/fields/v22/search_term_view
- CampaignSearchTermView fields: https://developers.google.com/google-ads/api/fields/v22/campaign_search_term_view
