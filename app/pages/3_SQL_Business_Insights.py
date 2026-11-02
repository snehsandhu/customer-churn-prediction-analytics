"""Phase 6 sql page."""
from churn_analytics.dashboard.components import run_page
from churn_analytics.dashboard.views.sql import render

run_page(render)
