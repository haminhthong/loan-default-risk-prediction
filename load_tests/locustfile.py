"""
Locust Performance & Load Testing Script for Loan Default Risk Prediction API.

Mục tiêu kiểm thử tải (Acceptance Criteria):
- Concurrency: 100 người dùng đồng thời
- Latency p95: < 500 ms
- Error rate: < 1.0%
- Throughput: >= 50 requests/sec

Cách chạy:
    locust -f load_tests/locustfile.py --headless -u 100 -r 10 --run-time 2m --host http://localhost:8000
"""

from locust import HttpUser, between, task

VALID_RECORD = {
    "loan_amnt": 10000.0,
    "term": "36 months",
    "installment": 334.54,
    "grade": "B",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 60000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "dti": 15.2,
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 1.0,
    "open_acc": 10.0,
    "pub_rec": 0.0,
    "revol_bal": 5000.0,
    "revol_util": "45.20%",
    "total_acc": 20.0,
    "earliest_cr_line": "Jan-00",
    "issue_d": "Dec-11",
}


class LoanAPIUser(HttpUser):
    """Giả lập người dùng gửi request chấm điểm khoản vay tới API endpoint."""

    wait_time = between(0.1, 0.5)

    @task(3)
    def score_single(self):
        """Kiểm thử endpoint /score với 1 hồ sơ."""
        self.client.post(
            "/score",
            json={"records": [VALID_RECORD]},
            name="/score (single record)",
        )

    @task(1)
    def score_batch(self):
        """Kiểm thử endpoint /score với batch 10 hồ sơ."""
        self.client.post(
            "/score",
            json={"records": [VALID_RECORD] * 10},
            name="/score (batch 10 records)",
        )

    @task(1)
    def health_check(self):
        """Kiểm thử endpoint /health."""
        self.client.get("/health", name="/health")
