# base.py
# Declarative base every ORM model inherits from.
# Single source of truth so app/db/session.py's create_all() can discover
# every model — as long as the model module has been imported somewhere
# before create_all() runs (see main.py hook below).

from sqlalchemy.orm import declarative_base

Base = declarative_base()