"""hand-curated role labels for our 50-person enron corpus subset.

source: public enron-corporate-history references (Klimt-Yang, Diesner et al.,
press coverage). roles assigned to the people who appear in our top-50 emailers
based on public job titles at enron in 1999-2002. people whose role is unclear
or external get None.

claim under test: graph-JEPA's frozen encoder, trained only with mask-and-predict
on the email graph, organizes people by FUNCTIONAL ROLE — lawyers near lawyers,
traders near traders, etc. — even though role labels never appeared in training.
this is non-tautological because role isn't a function of email volume alone:
lawyers email everyone (contract reviews), traders email each other and external
counterparties, executives have varied patterns.
"""

from __future__ import annotations

# email → functional role (from public enron records)
EMAIL_TO_ROLE: dict[str, str] = {
    # legal / counsel
    "sara.shackleton@enron.com": "legal",
    "mark.taylor@enron.com": "legal",
    "kay.mann@enron.com": "legal",
    "stephanie.panus@enron.com": "legal",
    "mary.hain@enron.com": "legal",
    "gerald.nemec@enron.com": "legal",
    "mary.cook@enron.com": "legal",
    "sarah.novosel@enron.com": "legal",
    # government / regulatory affairs
    "jeff.dasovich@enron.com": "govaffairs",
    "james.steffes@enron.com": "govaffairs",
    "alan.comnes@enron.com": "govaffairs",
    "susan.mara@enron.com": "govaffairs",
    "christi.nicolay@enron.com": "govaffairs",
    "richard.shapiro@enron.com": "govaffairs",
    # trading desk
    "eric.bass@enron.com": "trading",
    "kate.symes@enron.com": "trading",
    "mike.grigsby@enron.com": "trading",
    "chris.germany@enron.com": "trading",
    "david.forster@enron.com": "trading",
    # senior management
    "john.lavorato@enron.com": "exec",
    "louise.kitchen@enron.com": "exec",
    "sally.beck@enron.com": "exec",
    # admin / assistants / support
    "veronica.espinoza@enron.com": "admin",
    "rhonda.denton@enron.com": "admin",
    "cheryl.johnson@enron.com": "admin",
    "ginger.dernehl@enron.com": "admin",
    "lorna.brennan@enron.com": "admin",
    "janette.elbertson@enron.com": "admin",
    "julie.clyatt@enron.com": "admin",
    "miyung.buster@enron.com": "admin",
    "janet.butler@enron.com": "admin",
    "rosalee.fleming@enron.com": "admin",
    "taffy.milligan@enron.com": "admin",
    "tamara.black@enron.com": "admin",
    "susan.bailey@enron.com": "admin",
    # research
    "vince.kaminski@enron.com": "research",
    # operations
    "tana.jones@enron.com": "operations",  # legal specialist embedded with trading ops
    "pete.davis@enron.com": "operations",
}


def label_persons(email_list: list[str]) -> tuple[list[str], list[str]]:
    """returns (labels, unmapped) parallel to email_list. unmapped emails get
    label 'unknown'."""
    labels = []
    unmapped = []
    for e in email_list:
        r = EMAIL_TO_ROLE.get(e.strip().lower())
        if r is None:
            labels.append("unknown")
            unmapped.append(e)
        else:
            labels.append(r)
    return labels, unmapped
