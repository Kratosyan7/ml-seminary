import os
import smtplib
from email.message import EmailMessage
from typing import List

from .schemas import EmailDraft


class SMTPEmailSender:
    def __init__(self):
        self.host = os.getenv("SMTP_HOST", "")
        self.port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("SMTP_FROM_EMAIL", self.username)
        self.use_tls = os.getenv("SMTP_USE_TLS", "1") == "1"
        self.use_ssl = os.getenv("SMTP_USE_SSL", "0") == "1"

    def is_configured(self) -> bool:
        return bool(self.host and self.port and self.from_email)

    def send(self, draft: EmailDraft) -> None:
        if not self.is_configured():
            raise RuntimeError("SMTP не настроен. Нужны SMTP_HOST, SMTP_PORT, SMTP_FROM_EMAIL и при необходимости SMTP_USERNAME/SMTP_PASSWORD.")
        if not draft.to:
            raise RuntimeError(f"Нет адресатов для doc_id={draft.doc_id}")

        msg = EmailMessage()
        msg["From"] = self.from_email
        msg["To"] = ", ".join(draft.to)
        msg["Subject"] = draft.subject
        msg.set_content(draft.body)

        if self.use_ssl:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=30) as server:
                if self.username:
                    server.login(self.username, self.password)
                server.send_message(msg)
            return

        with smtplib.SMTP(self.host, self.port, timeout=30) as server:
            if self.use_tls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.send_message(msg)
