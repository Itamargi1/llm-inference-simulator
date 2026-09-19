"""Deterministic generator for the synthetic prompt dataset.

Why this exists
---------------
The simulator needs a workload that *looks* like real traffic to an LLM
service: a mix of task types and, more importantly, a wide spread of prompt
sizes. Prompt size drives simulated prefill cost and therefore queueing
behavior, so a dataset where every prompt is the same length would make the
simulation uninteresting.

Determinism
-----------
Everything random goes through a single `random.Random(SEED)` instance and the
generation order is fixed. Re-running this script produces a byte-identical
`data/prompts.json` unless the code or SEED is changed on purpose.

No external API or LLM is used - only the standard library.

Usage:
    python scripts/generate_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260918
TOTAL_RECORDS = 750
FIRST_ID = 1

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts.json"

CATEGORIES = ("summarization", "document_understanding", "customer_support")
TIERS = ("short", "medium", "long")

# How many "content units" (bullet lines, transcript turns, document rows...)
# a prompt body gets per length tier. These ranges are what actually produce
# the short/medium/long spread in the dataset.
UNITS_PER_TIER = {
    "short": (1, 2),
    "medium": (4, 7),
    "long": (12, 20),
    "xlong": (18, 26),
}

# Share of the "long" records promoted to the long-context tail. One in four
# long records gives roughly 8 percent of the dataset above 4,000 characters,
# which is enough to show prefill cost and KV-cache pressure growing with
# context without distorting the overall distribution.
XLONG_PROMOTE_EVERY = 4

# --------------------------------------------------------------------------
# Domains: reusable business contexts.
#
# Every text template below is formatted against one of these, so the same
# template produces visibly different prompts depending on the domain drawn.
# This is what keeps 750 records from looking like two templates with the
# names swapped.
# --------------------------------------------------------------------------

DOMAINS = [
    {
        "company": "Northwind Retail",
        "product": "Atlas Billing",
        "team": "Billing Platform",
        "feature": "invoice export",
        "metric": "invoice generation latency",
        "persona": "finance operations",
    },
    {
        "company": "Kestrel Logistics",
        "product": "RouteGrid",
        "team": "Dispatch Systems",
        "feature": "route optimization",
        "metric": "route solve time",
        "persona": "warehouse operations",
    },
    {
        "company": "Lumen Health",
        "product": "CareSync",
        "team": "Clinical Integrations",
        "feature": "appointment sync",
        "metric": "sync lag",
        "persona": "clinic administration",
    },
    {
        "company": "Vireo Analytics",
        "product": "Signalboard",
        "team": "Data Platform",
        "feature": "dashboard refresh",
        "metric": "dashboard load time",
        "persona": "business intelligence",
    },
    {
        "company": "Harbor Financial",
        "product": "Ledgerline",
        "team": "Payments",
        "feature": "reconciliation run",
        "metric": "settlement delay",
        "persona": "treasury",
    },
    {
        "company": "Peregrine Media",
        "product": "Streamdeck",
        "team": "Media Delivery",
        "feature": "transcoding pipeline",
        "metric": "time to first frame",
        "persona": "content operations",
    },
    {
        "company": "Ironwood Manufacturing",
        "product": "ShopFloor",
        "team": "Industrial IoT",
        "feature": "sensor ingestion",
        "metric": "telemetry ingest lag",
        "persona": "plant maintenance",
    },
    {
        "company": "Alder HR",
        "product": "Onboardly",
        "team": "People Tools",
        "feature": "onboarding checklist",
        "metric": "checklist completion rate",
        "persona": "human resources",
    },
    {
        "company": "Quillfoot Education",
        "product": "Coursemap",
        "team": "Learning Platform",
        "feature": "grade import",
        "metric": "import processing time",
        "persona": "academic administration",
    },
    {
        "company": "Basalt Energy",
        "product": "GridWatch",
        "team": "Grid Telemetry",
        "feature": "outage detection",
        "metric": "detection delay",
        "persona": "field operations",
    },
    {
        "company": "Marlowe Travel",
        "product": "Itinerae",
        "team": "Booking Services",
        "feature": "fare search",
        "metric": "search response time",
        "persona": "travel support",
    },
    {
        "company": "Sable Insurance",
        "product": "ClaimPath",
        "team": "Claims Automation",
        "feature": "claim intake",
        "metric": "intake turnaround",
        "persona": "claims adjusting",
    },
    {
        "company": "Fernwood Grocery",
        "product": "StockPulse",
        "team": "Inventory Systems",
        "feature": "stock forecasting",
        "metric": "forecast refresh time",
        "persona": "store operations",
    },
    {
        "company": "Crestline Telecom",
        "product": "Linkbase",
        "team": "Network Provisioning",
        "feature": "circuit provisioning",
        "metric": "provisioning duration",
        "persona": "network engineering",
    },
]

PLANS = ["Starter", "Growth", "Business", "Enterprise", "Legacy Pro"]

# --------------------------------------------------------------------------
# Content pools
#
# Kept as flat lists of format strings. Mixing several pools inside one prompt
# is what gives long prompts real internal structure instead of one repeated
# sentence.
# --------------------------------------------------------------------------

ENGINEERING_LINES = [
    "The {team} team shipped the {feature} rewrite behind a feature flag and enabled it for ten percent of accounts.",
    "{metric} dropped from 4.2 seconds to 1.8 seconds after the new query plan landed.",
    "We are still seeing intermittent timeouts when {product} retries against the read replica.",
    "Two engineers are on leave next week, so the migration work slips to the following sprint.",
    "The backfill job processed 3.1 million rows overnight and finished ahead of schedule.",
    "QA found a regression where {feature} silently drops records larger than 5 MB.",
    "The on-call rotation was noisy this week, mostly low-severity disk alerts.",
    "We agreed to freeze schema changes until the {product} upgrade completes.",
    "Staging cluster cost grew 18 percent month over month and needs a review.",
    "Documentation for {feature} is out of date and is blocking the {persona} team.",
    "The load test peaked at 4,800 requests per second before error rates climbed.",
    "A customer escalation forced us to reprioritize the caching work.",
    "Three flaky integration tests were quarantined so the pipeline stops blocking merges.",
    "The vendor confirmed the rate limit increase takes effect at the start of next month.",
    "Onboarding for the two new engineers is taking longer than planned because of missing environment access.",
    "The {persona} team asked whether {feature} can be scheduled weekly instead of daily.",
    "We reduced the container image size by 40 percent, which cut deploy time noticeably.",
    "A dependency upgrade pulled in a breaking change and was reverted the same afternoon.",
    "Read traffic is now served from the cache for 82 percent of {feature} requests.",
    "The data retention job has not run since the cluster upgrade and needs investigation.",
    "We are considering splitting the monolithic worker into per-queue processes.",
    "An audit flagged that service accounts have broader permissions than necessary.",
]

DECISION_LINES = [
    "Decision: roll {feature} out to all accounts on the 14th if error rates stay below 0.5 percent.",
    "Decision: keep the legacy endpoint available for one more quarter.",
    "Decision: move the nightly job to 03:00 UTC to avoid overlapping with the backup window.",
    "Action item: {team} to publish a rollback plan before the next release.",
    "Action item: schedule a follow-up review with the {persona} stakeholders.",
    "Action item: add {metric} to the executive dashboard by Friday.",
    "Open question: do we need a separate rate limit for bulk {feature} requests?",
    "Open question: who owns the runbook once the {team} reorganization finishes?",
    "Owner: platform team, due end of month.",
]

INCIDENT_LINES = [
    "At 02:14 UTC, error rates for {product} rose from 0.2 percent to 31 percent.",
    "The on-call engineer acknowledged the page at 02:19 UTC and opened an incident channel.",
    "Initial suspicion was a bad deploy, but the most recent release had been live for six hours.",
    "Root cause was connection pool exhaustion triggered by a slow downstream dependency.",
    "Mitigation was to raise the pool size and restart the affected workers.",
    "Full recovery was confirmed at 03:47 UTC once {metric} returned to baseline.",
    "Roughly 12,000 requests failed during the window, affecting 340 accounts.",
    "No data was lost; failed requests were retried automatically by the client SDK.",
    "Follow-up: add an alert on pool saturation, which currently has no coverage.",
    "The runbook for {feature} was missing a step, which slowed diagnosis by about 15 minutes.",
    "Customer support received 58 tickets referencing the outage before the status page was updated.",
    "The status page was updated 26 minutes after the first alert, which is outside our target.",
    "A second, smaller spike occurred at 04:10 UTC when the workers were restarted in parallel.",
    "The dashboard for {metric} did not refresh during the incident, which misled the responders.",
    "Our automated failover did not trigger because the health check only tested the process, not the dependency.",
    "The downstream provider later confirmed a degraded node in the same availability zone.",
    "Two engineers joined the call within ten minutes; a third was paged but did not respond.",
    "We considered failing over to the secondary region but judged the risk too high mid-incident.",
    "Retry storms from the client SDK amplified the load by an estimated factor of three.",
    "Timeline reconstruction was slowed because worker logs are sampled at 10 percent.",
    "Action item: raise the health check to include a dependency probe before the next release.",
    "Action item: document the manual failover procedure and rehearse it during the next game day.",
]

FEEDBACK_LINES = [
    'A {persona} lead wrote: "{product} is finally fast enough for our nightly runs, but {feature} still times out on large accounts."',
    'One reviewer said: "Setup took twenty minutes instead of the two days our old tool needed."',
    'A long-time customer noted: "The new interface is cleaner, but I cannot find where the old bulk upload went."',
    'An admin commented: "Support answered within an hour, which is much better than last year."',
    'A trial user wrote: "Pricing is confusing. I could not tell which plan includes {feature}."',
    'A {persona} manager said: "We still export everything to a spreadsheet because the built-in reports lack totals."',
    'One respondent wrote: "Mobile access is the main gap for us; the desktop experience is solid."',
    'A team lead noted: "Permissions are too coarse. Everyone who can view {feature} can also edit it."',
    'A customer replied: "The migration guide was accurate, which is rare. No surprises."',
    'A reviewer complained: "Notifications arrive hours late, so we rely on manual checks instead."',
    'An operations lead wrote: "We need audit history on {feature}. Right now we cannot tell who changed what."',
    'A new customer said: "Onboarding was smooth, but nobody told us about the API rate limits until we hit them."',
    'A {persona} analyst noted: "The search is too literal. If I misspell a name I get nothing back."',
    'One account manager wrote: "Renewal was painless, though the invoice arrived two weeks late."',
    'A user commented: "Dark mode looks good but several charts are unreadable in it."',
    'A department head said: "Training the team took longer than expected because the docs assume prior knowledge."',
    'A power user wrote: "I want scheduled exports. Doing this manually every Monday is wearing thin."',
    'A customer observed: "Performance is inconsistent. Some days {feature} is instant, other days it stalls."',
]

TRANSCRIPT_CUSTOMER = [
    "Customer: We have been waiting three days for the {feature} results and nothing has arrived.",
    "Customer: I was told this would be fixed after the last update, but it is the same.",
    "Customer: Our {persona} team cannot close the month until this is resolved.",
    "Customer: Is this affecting everyone or just our account?",
    "Customer: We are paying for the {plan} plan, so I expected better reliability than this.",
    "Customer: Can you just tell me roughly when it will work again?",
    "Customer: I already sent the logs to your colleague last week.",
    "Customer: If this is not fixed by Friday we will have to look at alternatives.",
    "Customer: That workaround helps, but it is not something we can do every day.",
    "Customer: Fine, I can try that now while we are on the call.",
    "Customer: Nobody told us the maintenance window would affect {feature} as well.",
    "Customer: We have three people sitting idle because of this.",
    "Customer: I do not have the ticket number in front of me, but it was opened last Monday.",
    "Customer: Does this mean the data we exported yesterday is incomplete?",
    "Customer: I would rather not reinstall everything again, we did that last time.",
    "Customer: Can you send me that in writing so I can forward it to my manager?",
    "Customer: We have been a customer for four years and this is the worst it has been.",
    "Customer: Understood. What should I do if it happens again over the weekend?",
]

TRANSCRIPT_AGENT = [
    "Agent: I understand, let me pull up the account and check the recent {feature} jobs.",
    "Agent: I can see three failed runs on your account, all with the same timeout error.",
    "Agent: This is not a global outage, it appears specific to accounts with large data volumes.",
    "Agent: I am going to escalate this to the {team} team with the logs attached.",
    "Agent: As a temporary workaround, you can split the export into smaller date ranges.",
    "Agent: I will not promise a date, but I will make sure you get an update by tomorrow.",
    "Agent: Thank you for your patience, I know this has taken longer than it should have.",
    "Agent: I have added a note to your ticket so you will not need to explain it again.",
    "Agent: Can you confirm whether the issue happens on every attempt or only some?",
    "Agent: I am raising the priority on this ticket given the month-end deadline.",
    "Agent: Let me check whether the maintenance window last night is related to this.",
    "Agent: The export you ran yesterday completed, so that data should be complete.",
    "Agent: You will not need to reinstall anything this time, this looks like a server-side issue.",
    "Agent: I will send you a written summary of what we discussed after the call.",
    "Agent: If it happens again over the weekend, reply to the ticket and it will page the on-call engineer.",
    "Agent: I am sorry, four years is a long time and you should not be dealing with this.",
    "Agent: Our engineering team deployed a fix to staging this morning, production is planned for Thursday.",
    "Agent: One moment, I want to verify this against another account on the same {plan} plan.",
]

RELEASE_LINES = [
    "Added: bulk actions for {feature}, available on {plan} plans and above.",
    "Added: a new API endpoint for querying {metric} over a custom date range.",
    "Changed: the default page size for list endpoints is now 50 instead of 25.",
    "Changed: {product} now retries failed webhooks for up to 24 hours.",
    "Fixed: timezone handling in scheduled reports, which previously used server time.",
    "Fixed: a crash when uploading files containing non-ASCII characters in the filename.",
    "Fixed: duplicate notifications when a record was edited twice within one minute.",
    "Deprecated: the v1 {feature} endpoint will be removed at the end of the quarter.",
    "Security: session tokens now expire after 12 hours of inactivity.",
    "Performance: {metric} improved by roughly 40 percent for accounts with over 100,000 records.",
    "Added: audit history showing who changed a {feature} configuration and when.",
    "Added: scheduled exports, configurable daily or weekly, with delivery to email or S3.",
    "Changed: search now tolerates minor spelling differences instead of requiring exact matches.",
    "Changed: the {feature} job queue is now processed in priority order rather than strictly first-in-first-out.",
    "Fixed: several charts were unreadable in dark mode due to insufficient contrast.",
    "Fixed: the account switcher occasionally showed workspaces the user could not access.",
    "Deprecated: password-only authentication for API access; use API keys or SSO instead.",
    "Known issue: very large {feature} jobs may still time out; a fix is planned for next month.",
]

SUMMARIZE_INSTRUCTIONS = [
    "Summarize the following {source} in three bullet points.",
    "Write a two-sentence summary of the {source} below.",
    "Give me a short executive summary of the {source} below.",
    "Summarize the {source} below and list any action items separately.",
    "Condense the following {source} into a paragraph a manager could read in under a minute.",
    "Produce a TL;DR of the {source} below, focusing on what changed and what is still open.",
    "Read the {source} and summarize the main risks it raises.",
    "Summarize the {source} below for someone who missed the discussion entirely.",
    "Extract the key decisions from the {source} below and summarize them.",
    "Summarize the {source} and note anything that requires a follow-up from the {persona} team.",
]

# --- Document understanding -----------------------------------------------

POLICY_CLAUSES = [
    "Section 3.1 Eligibility: Employees become eligible for the travel allowance after 90 days of continuous employment.",
    "Section 3.2 Approval: Any expense above 500 USD requires written approval from a department head before it is incurred.",
    "Section 4.1 Submission: Expense claims must be submitted within 30 days of the transaction date.",
    "Section 4.2 Receipts: Claims over 75 USD require an itemized receipt; card statements alone are not accepted.",
    "Section 5.1 Accommodation: Standard room rates are reimbursed up to 220 USD per night in listed metropolitan areas.",
    "Section 5.3 Meals: A per-diem of 65 USD applies on travel days and does not require receipts.",
    "Section 6.1 Mileage: Personal vehicle use is reimbursed at 0.62 USD per mile, excluding regular commuting.",
    "Section 7.2 Exceptions: Exceptions must be documented in writing and retained for seven years.",
    "Section 8.1 Non-compliance: Repeated late submissions may result in the claim being rejected entirely.",
    "Section 9.4 Currency: Foreign currency expenses are converted at the rate published on the transaction date.",
    "Section 3.4 Advance payments: Travel advances may be requested up to 14 days before departure and must be settled within 10 days of return.",
    "Section 5.2 Transport: Economy class is the standard for flights under six hours; premium economy requires director approval.",
    "Section 5.4 Entertainment: Client entertainment requires the names and affiliations of all attendees to be recorded.",
    "Section 6.2 Parking and tolls: These are reimbursed at cost and do not count toward the mileage allowance.",
    "Section 7.1 Personal expenses: Minibar charges, in-flight purchases and personal phone calls are not reimbursable.",
    "Section 8.3 Audit: A random sample of ten percent of claims is audited each quarter by the finance team.",
    "Section 10.1 Amendments: This policy is reviewed annually and supersedes all previous versions.",
]

SLA_CLAUSES = [
    "Clause 2.1: The Service Provider guarantees 99.9 percent monthly uptime measured at the load balancer.",
    "Clause 2.3: Scheduled maintenance windows are excluded from uptime calculations if announced 72 hours in advance.",
    "Clause 3.1: Severity 1 incidents receive a response within 30 minutes, 24 hours a day.",
    "Clause 3.2: Severity 3 incidents receive a response within one business day.",
    "Clause 4.2: Service credits are capped at 15 percent of the monthly fee for any single billing period.",
    "Clause 4.4: Credits must be requested in writing within 30 days of the incident.",
    "Clause 5.1: Data is retained for 90 days after termination, after which it is permanently deleted.",
    "Clause 6.3: Either party may terminate for convenience with 60 days written notice.",
    "Clause 7.1: Liability is limited to fees paid in the twelve months preceding the claim.",
    "Clause 8.2: The Provider may subcontract processing provided equivalent security controls apply.",
    "Clause 2.4: Uptime is calculated monthly and rounded to two decimal places, excluding excused downtime.",
    "Clause 3.3: Severity 2 incidents receive a response within four hours during business hours.",
    "Clause 4.1: Service credits are the sole and exclusive remedy for failure to meet the uptime commitment.",
    "Clause 5.2: The Customer may request a full data export at any time in a machine-readable format.",
    "Clause 6.1: This agreement renews automatically for successive twelve-month terms unless notice is given.",
    "Clause 7.3: Neither party is liable for indirect, incidental or consequential damages.",
    "Clause 9.1: The Provider shall notify the Customer of any confirmed data breach within 72 hours.",
    "Clause 10.2: Disputes shall first be escalated to named executives before formal proceedings begin.",
]

SPEC_CLAUSES = [
    "Requirement FR-12: The system shall accept {feature} submissions of up to 50 MB per request.",
    "Requirement FR-18: Failed submissions shall be retried up to three times with exponential backoff.",
    "Requirement NFR-04: 95th percentile response time shall not exceed 800 milliseconds under nominal load.",
    "Requirement NFR-07: The service shall sustain 1,200 concurrent sessions without degradation.",
    "Requirement SEC-02: All data in transit shall use TLS 1.2 or higher.",
    "Requirement SEC-05: Administrative actions shall be recorded in an append-only audit log.",
    "Requirement INT-03: The {feature} module shall expose a webhook on completion and on failure.",
    "Requirement OPS-01: The service shall emit {metric} as a metric at 15 second resolution.",
    "Requirement DAT-06: Records older than seven years shall be archived to cold storage automatically.",
    "Requirement UX-09: Users shall be able to cancel a running {feature} job from the interface.",
    "Requirement FR-21: The system shall validate submissions against the published schema before queuing them.",
    "Requirement FR-27: Partial failures within a batch shall not abort the remaining items in that batch.",
    "Requirement NFR-11: The service shall recover from a single node failure without operator intervention.",
    "Requirement NFR-15: Planned maintenance shall not require more than five minutes of write downtime.",
    "Requirement SEC-08: Secrets shall be retrieved from the managed vault at runtime and never written to disk.",
    "Requirement INT-07: The service shall expose a health endpoint returning dependency status.",
    "Requirement DAT-02: Every record shall carry a created timestamp and a last modified timestamp in UTC.",
    "Requirement OPS-05: Deployments shall support rollback to the previous version within ten minutes.",
]

INVOICE_LINES = [
    "Line 1 | {product} {plan} subscription | 12 months | 1,450.00 USD",
    "Line 2 | Additional user seats | 14 seats | 966.00 USD",
    "Line 3 | Premium support package | 12 months | 2,400.00 USD",
    "Line 4 | Data migration services | 38 hours | 5,700.00 USD",
    "Line 5 | Overage charges for {feature} | 214,000 records | 428.00 USD",
    "Line 6 | Onboarding and training | 2 sessions | 1,200.00 USD",
    "Line 7 | Sandbox environment | 12 months | 600.00 USD",
    "Line 8 | Discount, multi-year commitment | -1,340.00 USD",
    "Line 9 | Regional tax | 8.25 percent | 913.47 USD",
    "Line 10 | Dedicated customer success manager | 12 months | 3,600.00 USD",
    "Line 11 | Additional storage | 2 TB | 480.00 USD",
    "Line 12 | Custom report development | 16 hours | 2,400.00 USD",
    "Line 13 | Single sign-on add-on | 12 months | 900.00 USD",
    "Line 14 | Late payment fee, invoice from prior period | 125.00 USD",
]

REPORT_LINES = [
    "Q3 revenue for the {persona} segment reached 4.18 million USD, up 9 percent year over year.",
    "Churn among {plan} accounts was 2.1 percent, below the 3.0 percent target.",
    "Average {metric} across all regions was 1.9 seconds, against a stated objective of 2.5 seconds.",
    "Support ticket volume rose 14 percent, driven mainly by questions about {feature}.",
    "Net new accounts totaled 212, of which 41 upgraded within the first 60 days.",
    "Infrastructure spend was 312,000 USD, roughly 7 percent above the approved budget.",
    "The {team} team closed 148 of 173 planned work items during the quarter.",
    "Two regions missed their uptime objective, both due to a shared dependency failure.",
    "Expansion revenue from existing accounts accounted for 38 percent of total growth.",
    "Gross margin held at 74 percent despite the increase in infrastructure spend.",
    "The {persona} segment now represents 31 percent of total recurring revenue.",
    "Median time to first value for new accounts fell from 19 days to 12 days.",
    "Headcount grew by 14, with the {team} team accounting for six of those hires.",
    "Renewal rate by value was 106 percent, driven by upgrades on {plan} accounts.",
    "Three enterprise deals slipped from Q3 into Q4, representing roughly 480,000 USD.",
    "Marketing-sourced pipeline covered 2.4 times the quarterly target, above the 2.0 benchmark.",
]

DOC_TYPES = [
    ("expense policy", POLICY_CLAUSES),
    ("service level agreement", SLA_CLAUSES),
    ("technical specification", SPEC_CLAUSES),
    ("invoice", INVOICE_LINES),
    ("quarterly report", REPORT_LINES),
]

# Each question is paired with the clause(s) required to answer it, identified
# by a unique substring of the clause template.
#
# Why: clauses and questions used to be sampled independently, which regularly
# produced prompts asking about a section the document did not contain. Those
# are not useful test data - they are accidents. Generation now selects the
# questions first, guarantees their supporting clauses are present, and fills
# the rest of the document with distractors.
#
# An empty tuple means the question is answerable from whatever clauses appear
# (for example "which line item is largest").
DOC_QUESTION_SPECS = {
    "expense policy": [
        ("Does a 90 USD taxi fare need an itemized receipt?", ("Section 4.2",)),
        (
            "An employee submits a claim 45 days after the transaction. Is it still valid?",
            ("Section 4.1",),
        ),
        ("What is the reimbursement rate for using a personal vehicle?", ("Section 6.1",)),
        ("What approval is required for an expense of 600 USD?", ("Section 3.2",)),
        ("How long must documented exceptions be retained?", ("Section 7.2",)),
        ("Are minibar charges reimbursable?", ("Section 7.1",)),
        ("What is the meal per-diem on travel days?", ("Section 5.3",)),
        (
            "Do parking fees count toward the mileage allowance?",
            ("Section 6.2", "Section 6.1"),
        ),
    ],
    "service level agreement": [
        (
            "If uptime is 99.4 percent this month, what is the maximum credit the customer can claim?",
            ("Clause 2.1", "Clause 4.2"),
        ),
        (
            "A Severity 1 incident is reported at 23:50 on a Saturday. When is a response due?",
            ("Clause 3.1",),
        ),
        (
            "How much notice is required for maintenance to be excluded from uptime?",
            ("Clause 2.3",),
        ),
        ("What happens to customer data 120 days after termination?", ("Clause 5.1",)),
        ("Is the provider allowed to use subcontractors for processing?", ("Clause 8.2",)),
        (
            "Within what period must a confirmed data breach be reported?",
            ("Clause 9.1",),
        ),
        ("How much notice is required to terminate for convenience?", ("Clause 6.3",)),
    ],
    "technical specification": [
        (
            "What is the maximum accepted submission size, and how many retries are specified?",
            ("Requirement FR-12", "Requirement FR-18"),
        ),
        (
            "Which requirements relate to security, and what do they mandate?",
            ("Requirement SEC-02", "Requirement SEC-05"),
        ),
        (
            "Does the specification require the job to be cancellable by the user?",
            ("Requirement UX-09",),
        ),
        ("What resolution is required for metric emission?", ("Requirement OPS-01",)),
        ("Is there a stated concurrency target, and what is it?", ("Requirement NFR-07",)),
        (
            "How quickly must a deployment be able to roll back?",
            ("Requirement OPS-05",),
        ),
        (
            "What is the stated 95th percentile response time budget?",
            ("Requirement NFR-04",),
        ),
    ],
    "invoice": [
        ("What is the total before tax?", ("Line 9 |",)),
        (
            "Which line items are one-time charges rather than recurring subscriptions?",
            ("Line 1 |", "Line 4 |"),
        ),
        ("How much was the discount, and what was it for?", ("Line 8 |",)),
        ("What is the per-seat cost of the additional user seats?", ("Line 2 |",)),
        ("Which single line item is the largest, and what does it cover?", ()),
        ("Was any late payment fee charged, and how much?", ("Line 14 |",)),
    ],
    "quarterly report": [
        ("Did the quarter meet its churn target?", ("Churn among",)),
        (
            "Which metrics came in worse than their stated objective?",
            ("Infrastructure spend was", "Two regions missed their uptime objective"),
        ),
        ("What drove the increase in support ticket volume?", ("Support ticket volume rose",)),
        (
            "How much of the growth came from existing accounts rather than new ones?",
            ("Expansion revenue from existing accounts",),
        ),
        ("By how much did infrastructure spend exceed budget?", ("Infrastructure spend was",)),
        ("Did average response time meet its objective?", ("across all regions was",)),
    ],
}

DOC_INSTRUCTIONS = [
    "Using only the {doc_type} below, answer the question that follows.",
    "Read the {doc_type} and answer the question at the end. Quote the relevant section.",
    "Based on the {doc_type} below, answer the question. If the document does not say, reply that it is not specified.",
    "Review this {doc_type} and answer the question underneath it.",
    "Answer the question at the bottom using the {doc_type} provided. Do not use outside knowledge.",
]

# --- Customer support ------------------------------------------------------

SUPPORT_ISSUES = [
    "I cannot log in to {product}. It keeps saying my password is incorrect even though I just reset it.",
    "How do I add a second administrator to our {product} account?",
    "Our {feature} report is empty this morning. Is something down?",
    "Can you tell me why we were charged twice this month?",
    "The mobile app crashes as soon as I open the {feature} screen.",
    "Where do I download last quarter's invoices?",
    "I need to change the billing email address on our account.",
    "Is there a way to export {feature} data as CSV?",
    "We hit a rate limit error and I do not understand which limit we exceeded.",
    "The password reset email never arrives, not even in the spam folder.",
    "A teammate was removed from the account by mistake. Can you restore their access?",
    "Does {product} support single sign-on with Okta?",
    "Our scheduled {feature} job ran twice last night and created duplicate records.",
    "The totals in {product} do not match what our accountant calculated.",
    "I upgraded to the {plan} plan but the new features are still locked.",
    "The {feature} page has been loading forever since this morning. Is anyone else reporting this?",
    "We need to cancel one of our two subscriptions before the renewal date. How do we do that?",
    "Our API integration started returning 403 errors overnight with no changes on our end.",
    "Can we get a copy of your security documentation for our vendor review?",
    "The notification emails from {product} are going to our spam folder. Can that be fixed?",
    "I set up a scheduled {feature} job but it never ran. There is no error shown anywhere.",
    "Two of our users see different numbers on the same {product} dashboard. Which one is right?",
    "We are migrating to a new domain. What happens to our existing user accounts?",
    "Is there an audit log showing who deleted records from our workspace last week?",
    "The search in {product} returns nothing for names I know exist in the system.",
]

SUPPORT_CONTEXT = [
    "Account ID: {acct}",
    "Plan: {plan}, billed annually",
    "Order number: {order}",
    "This started on the {day}th at around {hour}:00 UTC.",
    "We have about {users} active users on the account.",
    "Browser: Chrome 128 on Windows 11.",
    "Region: eu-west-1.",
    "Ticket reference from the previous conversation: {ticket}",
    "The affected workspace is the one named after our {persona} department.",
]

# Short tickets get exactly one identifying line. Every entry carries a
# high-cardinality identifier so that two short prompts built from the same
# issue template do not come out byte-identical.
SUPPORT_SHORT_DETAILS = [
    "Account ID: {acct}",
    "Order number: {order}",
    "Ticket reference: {ticket}",
    "Account {acct}, on the {plan} plan.",
    "Invoice in question: {invoice_no}",
    "Account ID {acct}; this started on the {day}th at around {hour}:00 UTC.",
]

SUPPORT_TROUBLESHOOTING = [
    "I already cleared the browser cache and tried an incognito window.",
    "We tested from two different networks and got the same result.",
    "Your colleague asked us to rotate the API key, which we did on Tuesday.",
    "I reinstalled the desktop client and the problem persists.",
    "We followed the workaround in your help article about {feature}, but it did not help.",
    "Disabling our ad blocker made no difference.",
    "Another admin tried from a different laptop and saw the same error.",
    "We confirmed with our IT team that nothing is blocked by the corporate firewall.",
    "I waited 24 hours as suggested, then tried again with the same outcome.",
]

SUPPORT_HISTORY = [
    "On the 3rd I opened ticket {ticket} and was told it was a known issue.",
    "On the 7th an agent replied asking for logs, which I attached the same day.",
    "On the 9th the ticket was marked resolved, but nothing had changed on our side.",
    "I replied to reopen it and did not hear back for four days.",
    "A different agent then suggested upgrading to the {plan} plan, which we had already done.",
    "We were promised a callback that never happened.",
    "The last update said engineering was investigating, but that was two weeks ago.",
]

LOG_TEMPLATES = [
    "2026-09-{day:02d}T{hour:02d}:14:03Z ERROR {product}.api request_id={req} status=503 message=upstream timeout",
    "2026-09-{day:02d}T{hour:02d}:14:04Z WARN  {product}.worker job={feature} attempt=2 backoff=4s",
    "2026-09-{day:02d}T{hour:02d}:15:11Z ERROR {product}.api request_id={req2} status=429 message=rate limit exceeded",
    "2026-09-{day:02d}T{hour:02d}:16:47Z INFO  {product}.auth user={acct} action=token_refresh result=ok",
    "2026-09-{day:02d}T{hour:02d}:17:02Z ERROR {product}.worker job={feature} attempt=3 result=failed",
]

SUPPORT_CLOSINGS = [
    "Could you tell me whether this is something we can fix ourselves or whether it needs your engineering team?",
    "What is the fastest path to getting this resolved? We have a deadline on Friday.",
    "Please let me know if you need anything else from our side.",
    "I would appreciate an update today, even if there is no fix yet.",
    "Can you confirm whether other customers are affected by the same problem?",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def take(rng: random.Random, pool: list[str], count: int) -> list[str]:
    """Draw `count` distinct entries from a pool, without exceeding its size.

    Distinct entries matter: sampling with replacement would produce long
    prompts that repeat the same sentence, which the dataset must avoid.
    """
    return rng.sample(pool, min(count, len(pool)))


def unit_count(rng: random.Random, tier: str) -> int:
    low, high = UNITS_PER_TIER[tier]
    return rng.randint(low, high)


def build_context(rng: random.Random) -> dict[str, object]:
    """One prompt's substitution context: a domain plus per-record details."""
    domain = rng.choice(DOMAINS)
    context: dict[str, object] = dict(domain)
    context.update(
        {
            "plan": rng.choice(PLANS),
            "acct": "ACC-{}".format(rng.randint(100000, 999999)),
            "order": "ORD-{}".format(rng.randint(10000, 99999)),
            "ticket": "SUP-{}".format(rng.randint(10000, 99999)),
            "invoice_no": "INV-2026-{}".format(rng.randint(1000, 9999)),
            "users": rng.choice([8, 12, 25, 40, 68, 120, 310]),
            "day": rng.randint(1, 28),
            "hour": rng.randint(0, 23),
            "req": "{:04x}".format(rng.randint(0x1000, 0xFFFF)),
            "req2": "{:04x}".format(rng.randint(0x1000, 0xFFFF)),
        }
    )
    return context


def fill(lines: list[str], context: dict[str, object]) -> list[str]:
    return [line.format(**context) for line in lines]


def find_clause(pool: list[str], key: str) -> str:
    """Return the clause template in `pool` identified by `key`.

    Raises at generation time if a question references a clause that does not
    exist, so a typo in the question/clause mapping cannot quietly produce
    unanswerable prompts.
    """
    for clause in pool:
        if key in clause:
            return clause
    raise KeyError(f"No clause matching {key!r} in pool.")


# --------------------------------------------------------------------------
# Parametric row generators.
#
# The long-context tail is built from these rather than from hundreds of extra
# static sentences. Every row varies by timestamp, identifier, amount and
# status, so a long prompt is genuinely long rather than one sentence repeated.
# --------------------------------------------------------------------------

LEDGER_DESCRIPTIONS = [
    "{feature} overage",
    "subscription true-up",
    "support retainer",
    "professional services",
    "storage expansion",
    "seat adjustment",
    "training session",
    "sandbox environment",
]

LEDGER_STATUSES = ["approved", "pending review", "posted", "disputed", "reconciled"]


def generate_ledger_rows(
    rng: random.Random, context: dict[str, object], count: int
) -> list[str]:
    """Transaction-style rows for long document appendices."""
    rows = []
    for _ in range(count):
        rows.append(
            "2026-{:02d}-{:02d} | TXN-{:06d} | {:<24} | {:>10,.2f} USD | {}".format(
                rng.randint(1, 12),
                rng.randint(1, 28),
                rng.randint(100000, 999999),
                rng.choice(LEDGER_DESCRIPTIONS).format(**context),
                rng.uniform(40.0, 9800.0),
                rng.choice(LEDGER_STATUSES),
            )
        )
    return rows


TIMELINE_ACTIONS = [
    "paged the secondary on-call engineer",
    "restarted the worker pool in the primary region",
    "confirmed the replica lag was still climbing",
    "opened a ticket with the upstream provider",
    "rolled back the configuration change from the previous evening",
    "drained traffic from two unhealthy nodes",
    "raised the connection pool limit to 200",
    "posted a customer-facing status update",
    "verified that queued jobs were not being dropped",
    "captured a heap dump from the affected process",
    "escalated to the database on-call rotation",
    "began replaying the failed requests from the dead letter queue",
]

TIMELINE_ACTORS = [
    "the on-call engineer",
    "the incident commander",
    "the {team} lead",
    "the database on-call engineer",
    "the support duty manager",
]


def generate_timeline_entries(
    rng: random.Random, context: dict[str, object], count: int
) -> list[str]:
    """Minute-by-minute incident timeline for long postmortems."""
    entries = []
    minute = rng.randint(5, 20)
    for _ in range(count):
        minute += rng.randint(1, 7)
        entries.append(
            "{:02d}:{:02d} UTC - {} {}.".format(
                2 + minute // 60,
                minute % 60,
                rng.choice(TIMELINE_ACTORS).format(**context),
                rng.choice(TIMELINE_ACTIONS).format(**context),
            )
        )
    return entries


LOG_COMPONENTS = ["api", "worker", "auth", "scheduler", "webhook"]

# Level, status and message are paired rather than drawn independently: an
# "INFO ... status=500" line would be obviously wrong to anyone reading the
# dataset.
LOG_EVENTS = [
    ("ERROR", 503, "upstream timeout"),
    ("ERROR", 500, "connection reset by peer"),
    ("ERROR", 500, "dependency unavailable"),
    ("ERROR", 413, "payload too large"),
    ("WARN ", 429, "rate limit exceeded"),
    ("WARN ", 503, "retry scheduled"),
    ("WARN ", 400, "malformed request body"),
    ("INFO ", 200, "token refresh ok"),
    ("INFO ", 202, "job accepted"),
    ("INFO ", 200, "request completed"),
]


def generate_log_lines(
    rng: random.Random, context: dict[str, object], count: int
) -> list[str]:
    """Application log excerpt for long support tickets."""
    lines = []
    second = rng.randint(0, 40)
    for _ in range(count):
        second += rng.randint(1, 30)
        level, status, message = rng.choice(LOG_EVENTS)
        lines.append(
            "2026-09-{:02d}T{:02d}:{:02d}:{:02d}Z {} {}.{} request_id={:04x} status={} message={}".format(
                context["day"],
                context["hour"],
                (second // 60) % 60,
                second % 60,
                level,
                context["product"],
                rng.choice(LOG_COMPONENTS),
                rng.randint(0x1000, 0xFFFF),
                status,
                message,
            )
        )
    return lines


# --------------------------------------------------------------------------
# Category builders
#
# Each returns a finished prompt string for the requested length tier.
# --------------------------------------------------------------------------


def build_summarization(rng: random.Random, tier: str, context: dict[str, object]) -> str:
    """Summarization: an instruction plus a body of source material.

    The source type is drawn per record so the dataset contains meeting notes,
    postmortems, transcripts, feedback batches and release notes rather than
    one shape of document.
    """
    if tier == "xlong":
        # Only the source types that can plausibly carry a long parametric
        # section (a timeline) are used for the long-context tail.
        source_kind = rng.choice(
            ["incident postmortem", "meeting notes", "weekly status update"]
        )
    else:
        source_kind = rng.choice(
            ["meeting notes", "incident postmortem", "support call transcript",
             "customer feedback batch", "weekly status update", "release notes"]
        )
    n = unit_count(rng, tier)

    if source_kind == "incident postmortem":
        body_lines = take(rng, INCIDENT_LINES, n)
        if tier in ("long", "xlong"):
            # Long postmortems pull from several pools. Drawing everything
            # from one pool would hit its size limit and cap the prompt
            # length, which is exactly the variation we need.
            body_lines += take(rng, DECISION_LINES, 4)
            body_lines += take(rng, ENGINEERING_LINES, 5)
        header = "Incident postmortem draft - {product}".format(**context)
        body = header + "\n" + "\n".join(fill(body_lines, context))
        if tier == "xlong":
            body += "\n\nDetailed response timeline:\n" + "\n".join(
                generate_timeline_entries(rng, context, rng.randint(35, 70))
            )

    elif source_kind == "support call transcript":
        # Interleave the two speakers so a transcript reads like a dialogue.
        turns = max(2, n)
        customer = take(rng, TRANSCRIPT_CUSTOMER, (turns + 1) // 2)
        agent = take(rng, TRANSCRIPT_AGENT, (turns + 1) // 2)
        interleaved: list[str] = []
        for c_line, a_line in zip(customer, agent):
            interleaved.append(c_line)
            interleaved.append(a_line)
        body = "\n".join(fill(interleaved[:turns], context))

    elif source_kind == "customer feedback batch":
        body_lines = take(rng, FEEDBACK_LINES, n)
        if tier == "long":
            body_lines += take(rng, REPORT_LINES, 4)
        body = "\n".join(fill(body_lines, context))

    elif source_kind == "release notes":
        body_lines = take(rng, RELEASE_LINES, n)
        if tier == "long":
            body_lines += take(rng, ENGINEERING_LINES, 5)
        header = "{product} release notes".format(**context)
        body = header + "\n" + "\n".join(fill(body_lines, context))

    else:  # meeting notes / weekly status update
        body_lines = take(rng, ENGINEERING_LINES, n)
        if tier == "medium":
            body_lines += take(rng, DECISION_LINES, 2)
        elif tier in ("long", "xlong"):
            body_lines += take(rng, DECISION_LINES, 6)
            body_lines += take(rng, INCIDENT_LINES, 4)
        header = "{team} - {source}".format(source=source_kind, **context)
        body = header + "\n" + "\n".join(fill(body_lines, context))
        if tier == "xlong":
            body += "\n\nOperational log reviewed during the meeting:\n" + "\n".join(
                generate_timeline_entries(rng, context, rng.randint(35, 70))
            )

    instruction = rng.choice(SUMMARIZE_INSTRUCTIONS).format(source=source_kind, **context)
    return instruction + "\n\n" + body


def build_document_understanding(
    rng: random.Random, tier: str, context: dict[str, object]
) -> str:
    """Document understanding: a document excerpt plus question(s) about it."""
    doc_type, clause_pool = rng.choice(DOC_TYPES)
    n = unit_count(rng, tier)

    # Questions are chosen first, then the clauses they depend on are pinned
    # into the document. Distractors fill whatever length remains.
    question_count = {"short": 1, "medium": 2, "long": 3, "xlong": 4}[tier]
    specs = take(rng, DOC_QUESTION_SPECS[doc_type], question_count)

    required: list[str] = []
    for _question, keys in specs:
        for key in keys:
            clause = find_clause(clause_pool, key)
            if clause not in required:
                required.append(clause)

    distractors = [clause for clause in clause_pool if clause not in required]
    clauses = required + take(rng, distractors, max(0, n - len(required)))
    # Shuffle so the answer-bearing clauses are not always at the top.
    rng.shuffle(clauses)

    # Long prompts pull in a second document section so the model has to
    # navigate more structure, not just more text.
    if tier in ("long", "xlong"):
        extra_type, extra_pool = rng.choice(DOC_TYPES)
        if extra_type != doc_type:
            clauses = clauses + ["", "Appendix A - extract from the {}:".format(extra_type)]
            clauses += take(rng, extra_pool, 7)

    if tier == "xlong":
        # A long transaction appendix: parametric rows, so length comes from
        # genuinely varied content rather than repetition.
        clauses += ["", "Appendix B - transaction ledger for the period:"]
        clauses += generate_ledger_rows(rng, context, rng.randint(45, 90))

    if doc_type == "invoice":
        header = "Invoice {invoice_no} - {company} - {product}".format(**context)
    else:
        header = "{} excerpt - {}".format(doc_type.title(), context["company"])

    questions = [question for question, _keys in specs]
    question_block = "\n".join(
        "Question {}: {}".format(i + 1, q) for i, q in enumerate(fill(questions, context))
    )

    instruction = rng.choice(DOC_INSTRUCTIONS).format(doc_type=doc_type, **context)
    document = header + "\n" + "\n".join(fill(clauses, context))
    return instruction + "\n\n" + document + "\n\n" + question_block


def build_customer_support(
    rng: random.Random, tier: str, context: dict[str, object]
) -> str:
    """Customer support: a user's message, with context growing by tier.

    Short prompts are a bare question. Medium adds account context. Long adds
    a history of prior contact, troubleshooting already attempted and a log
    excerpt - which is what makes real support tickets long.
    """
    issue = rng.choice(SUPPORT_ISSUES).format(**context)

    if tier == "short":
        # Even a brief ticket normally carries one identifying detail. It also
        # keeps short prompts distinct from one another, since several issue
        # templates contain no domain placeholders of their own.
        detail = rng.choice(SUPPORT_SHORT_DETAILS).format(**context)
        return issue + "\n" + detail

    parts = [issue, ""]

    if tier == "medium":
        parts += fill(take(rng, SUPPORT_CONTEXT, rng.randint(2, 3)), context)
        parts += fill(take(rng, SUPPORT_TROUBLESHOOTING, rng.randint(1, 2)), context)
    else:  # long / xlong
        parts += fill(take(rng, SUPPORT_CONTEXT, rng.randint(3, 5)), context)
        parts.append("")
        parts.append("What we have already tried:")
        parts += fill(take(rng, SUPPORT_TROUBLESHOOTING, rng.randint(3, 5)), context)
        parts.append("")
        parts.append("History of this issue:")
        parts += fill(take(rng, SUPPORT_HISTORY, rng.randint(3, 5)), context)
        parts.append("")
        parts.append("Relevant log lines from our side:")
        if tier == "xlong":
            # A realistic escalation attaches a full log excerpt rather than
            # five lines, which is what pushes these into the long-context tail.
            parts += generate_log_lines(rng, context, rng.randint(40, 80))
        else:
            parts += fill(take(rng, LOG_TEMPLATES, rng.randint(3, 5)), context)

    parts.append("")
    parts.append(rng.choice(SUPPORT_CLOSINGS).format(**context))
    return "\n".join(parts)


BUILDERS = {
    "summarization": build_summarization,
    "document_understanding": build_document_understanding,
    "customer_support": build_customer_support,
}


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------


def build_plan() -> list[tuple[str, str]]:
    """Fixed (category, tier) plan: balanced across categories and tiers.

    Built deterministically before any randomness is drawn, so the category
    and length distribution is exact rather than sampled.
    """
    plan: list[tuple[str, str]] = []
    per_category = TOTAL_RECORDS // len(CATEGORIES)
    for category in CATEGORIES:
        for index in range(per_category):
            plan.append((category, TIERS[index % len(TIERS)]))

    # Distribute any remainder (non-divisible totals) across the categories.
    remainder = TOTAL_RECORDS - len(plan)
    for index in range(remainder):
        plan.append((CATEGORIES[index % len(CATEGORIES)], TIERS[index % len(TIERS)]))

    # Promote a deterministic slice of the long records to the long-context
    # tail. Only "long" entries are touched, so the short and medium workloads
    # keep exactly the distribution they had.
    long_positions = [i for i, (_cat, tier) in enumerate(plan) if tier == "long"]
    for count, position in enumerate(long_positions):
        if count % XLONG_PROMOTE_EVERY == 0:
            plan[position] = (plan[position][0], "xlong")
    return plan


def generate_records() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    plan = build_plan()
    # Shuffle so categories are interleaved in the file rather than appearing
    # in three contiguous blocks; IDs are assigned after the shuffle.
    rng.shuffle(plan)

    records: list[dict[str, object]] = []
    for offset, (category, tier) in enumerate(plan):
        context = build_context(rng)
        prompt = BUILDERS[category](rng, tier, context)
        records.append(
            {
                "id": FIRST_ID + offset,
                "category": category,
                "prompt": prompt,
            }
        )

    # A workload made of repeated identical prompts would be a poor test of
    # the simulator, so duplicates are treated as a generation bug.
    prompts = [record["prompt"] for record in records]
    if len(set(prompts)) != len(prompts):
        raise RuntimeError(
            "Generated dataset contains duplicate prompts "
            f"({len(prompts) - len(set(prompts))} duplicates)."
        )
    return records


def main() -> None:
    records = generate_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys + fixed indent keeps the output byte-stable across runs, and
    # newline="\n" keeps it byte-stable across operating systems: without it
    # Windows would translate every newline to CRLF and the same dataset would
    # hash differently than on Linux or macOS.
    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("Wrote {} records to {}".format(len(records), OUTPUT_PATH))


if __name__ == "__main__":
    main()
