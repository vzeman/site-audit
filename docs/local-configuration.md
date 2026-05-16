# Local Configuration

Site Audit can read persistent defaults from a local `.env` file. The file is ignored by git, so it is the right place for API credentials and repeated run settings.

Command-line flags still win. The priority is:

1. explicit CLI flag
2. `.env` value
3. built-in default

## Naming Rules

Every CLI setting can be written as an environment variable.

For command-specific settings, use:

```text
SITE_AUDIT_<COMMAND>_<SETTING>
```

Examples:

```bash
SITE_AUDIT_RUN_DOMAIN="example.com"
SITE_AUDIT_RUN_MAX_PAGES="500"
SITE_AUDIT_RUN_WORKERS="4"
SITE_AUDIT_RUN_SEARCH_PROVIDER="google_ads"
SITE_AUDIT_RUN_COMPETITIVE_AUTO="true"
SITE_AUDIT_RUN_COMPETITIVE_AUTO_PRODUCT_SEED="help desk software, customer support software"
SITE_AUDIT_SERVE_DOMAIN="example.com"
SITE_AUDIT_SERVE_PORT="8765"
```

Global settings can omit the command:

```bash
SITE_AUDIT_VERBOSE="true"
```

Boolean values accept `true`, `false`, `1`, `0`, `yes`, `no`, `on`, and `off`.

Repeated values such as `--sitemap-url`, `--url-exclude`, and `--competitive-auto-product-seed` can be comma-separated or entered one per line in the settings screen.

## Local App

Open the local app:

```bash
site-audit settings
```

Then open:

```text
http://127.0.0.1:8780/
```

The top menu links to Reports, Comparisons, and Settings. Reports and
comparison dashboards are served from the same local app, so you do not
need a separate viewer just to open generated outputs.

The settings screen shows:

- every CLI setting
- the `.env` key used for that setting
- the built-in default
- help text
- credential fields for GSC, Ahrefs, DataForSEO, and Google Ads

Save writes to `.env` while preserving unrelated lines and comments.

Use another file or port:

```bash
site-audit settings --env-file .env.local --port 8790
```

## Example .env

```bash
# Repeated audit defaults
SITE_AUDIT_RUN_DOMAIN="example.com"
SITE_AUDIT_RUN_MAX_PAGES="500"
SITE_AUDIT_RUN_WORKERS="4"
SITE_AUDIT_RUN_NO_SNAPSHOT="true"

# Competitive paragraph-gap defaults
SITE_AUDIT_RUN_COMPETITIVE_AUTO="true"
SITE_AUDIT_RUN_COMPETITIVE_AUTO_CLUSTERS="5"
SITE_AUDIT_RUN_COMPETITIVE_AUTO_KEYWORDS_PER_CLUSTER="1"
SITE_AUDIT_RUN_COMPETITIVE_AUTO_RESULTS_PER_KEYWORD="5"
SITE_AUDIT_RUN_COMPETITIVE_AUTO_PRODUCT_SEED="help desk software, live chat software"

# Search provider. Use "all" to compare every enabled source in one report.
SITE_AUDIT_RUN_SEARCH_PROVIDER="all"
GOOGLE_ADS_CUSTOMER_ID="123-456-7890"
GOOGLE_ADS_DEVELOPER_TOKEN="..."
GOOGLE_ADS_CLIENT_ID="..."
GOOGLE_ADS_CLIENT_SECRET="..."
GOOGLE_ADS_REFRESH_TOKEN="..."
```

Now this is enough:

```bash
site-audit run
```

And this is enough to open the saved domain report:

```bash
site-audit serve
```
