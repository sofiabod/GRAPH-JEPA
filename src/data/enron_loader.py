import os
import tarfile
import urllib.request
from pathlib import Path
from email import message_from_string
from datetime import datetime


ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz"


def download_enron(data_dir):
    # downloads enron corpus to data_dir/raw/
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "enron_mail_20150507.tgz"
    if dest.exists():
        return
    urllib.request.urlretrieve(ENRON_URL, dest)
    with tarfile.open(dest, "r:gz") as tar:
        tar.extractall(raw_dir)


def load_emails(data_dir):
    # returns list of dicts with sender, recipients, timestamp, subject, body
    emails = []
    maildir = Path(data_dir) / "raw" / "maildir"
    if not maildir.exists():
        return emails
    for user_dir in sorted(maildir.iterdir()):
        for folder in user_dir.iterdir():
            if not folder.is_dir():
                continue
            for msg_file in folder.iterdir():
                if not msg_file.is_file():
                    continue
                try:
                    text = msg_file.read_text(errors="replace")
                    msg = message_from_string(text)
                    sender = msg.get("From", "").strip().lower()
                    to_raw = msg.get("To", "")
                    recipients = [r.strip().lower() for r in to_raw.split(",") if r.strip()]
                    date_str = msg.get("Date", "")
                    subject = msg.get("Subject", "")
                    body = msg.get_payload(decode=False) or ""
                    if isinstance(body, list):
                        body = ""
                    emails.append({
                        "sender": sender,
                        "recipients": recipients,
                        "date_str": date_str,
                        "subject": subject,
                        "body": str(body),
                    })
                except Exception:
                    continue
    return emails


def normalize_email(addr):
    # strip display name, lowercase, keep only enron.com addresses where possible
    addr = addr.strip().lower()
    if "<" in addr:
        addr = addr.split("<")[1].split(">")[0].strip()
    return addr
