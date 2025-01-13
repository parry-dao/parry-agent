"""Database engine configuration."""

from __future__ import annotations

import sqlite3
from typing import Final

from platformdirs import user_data_dir
from sqlmodel import create_engine
from upath import UPath


# App identifiers
APP_NAME: Final = "parry-agent"
APP_AUTHOR: Final = "parry"

# Default locations
DATA_DIR: Final = UPath(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_DB_NAME: Final = "history.db"


def get_database_path() -> UPath:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# Create engine with better SQLite settings
engine = create_engine(
    f"sqlite:///{get_database_path()}",
    connect_args={"check_same_thread": False},
    creator=lambda: sqlite3.connect(str(get_database_path()), check_same_thread=False),
)


def init_database():
    """Initialize database tables."""
    import sqlalchemy as sa
    from sqlalchemy import inspect

    from storage.models import SQLModel

    # Create tables if they don't exist
    SQLModel.metadata.create_all(engine)

    # Auto-add missing columns
    with engine.connect() as conn:
        inspector = inspect(engine)

        # For each table in our models
        for table_name, table in SQLModel.metadata.tables.items():
            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

            # For each column in model that doesn't exist in DB
            for col in table.columns:
                if col.name not in existing_columns:
                    # Create ALTER TABLE statement based on column type
                    type_sql = col.type.compile(engine.dialect)
                    nullable = "" if col.nullable else " NOT NULL"
                    default = ""
                    match col.default:
                        case None:
                            pass
                        case _ if hasattr(col.default, "arg"):
                            # Simple default value
                            default = f" DEFAULT {col.default.arg}"  # pyright: ignore
                        case sa.Computed():
                            # Computed default
                            default = f" DEFAULT {col.default.sqltext}"
                        case sa.FetchedValue():
                            # Server-side default
                            pass
                    text = sa.text(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {col.name} {type_sql}{nullable}{default}"
                    )
                    conn.execute(text)

        conn.commit()
