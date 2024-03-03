# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import os
from typing import List, Optional, Tuple
from src.config import config

import mlrun
import pandas as pd
from sqlalchemy import (
    ForeignKey,
    Boolean,
    Date,
    Enum,
    Float,
    Integer,
    String,
    Time,
    bindparam,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    mapped_column,
    sessionmaker,
)

MYSQL_URL = config.sql_connection_str
ID_LENGTH = 32
FILE_PATH_LENGTH = 500
Base = declarative_base()

class Account(Base):
    __tablename__ = "account"

    # Metadata:
    account_name: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("account.account_name"), primary_key=True
    )
    account_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("account.account_id")
    )
    # Analysis:
    password: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )
    balance: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )


def create_tables():
    """
    Create the table for when creating project.
    """
    # Create an engine:
    engine = create_engine(url=MYSQL_URL)

    # Create the schema's tables:
    Base.metadata.create_all(engine)


def insert_accounts(accounts: pd.DataFrame):
    # Create an engine:
    engine = create_engine(url=MYSQL_URL)

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Cast data from dataframe to a list of dictionaries:
    records = accounts.to_dict(orient="records")

    # Insert the new accounts into the table and commit:
    with session.begin() as sess:
        sess.execute(insert(Account), records)


def update_accounts(
    table_key: str,
    data_key: str,
    data: pd.DataFrame,
):
    # Create an engine:
    engine = create_engine(url=MYSQL_URL)

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Make sure keys are not duplicates (so we can update by the key with `bindparam`):
    if data_key == table_key:
        data_key += "_2"
        data.rename(columns={table_key: data_key}, inplace=True)

    # Cast data from dataframe to a list of dictionaries:
    data = data.to_dict(orient="records")

    # Insert the new accounts into the table and commit:
    with session.begin() as sess:
        sess.connection().execute(
            update(Account).where(getattr(Account, table_key) == bindparam(data_key)), data
        )


def get_accounts() -> pd.DataFrame:
    # Create an engine:
    engine = create_engine(url=MYSQL_URL)

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Select all calls:
    with session.begin() as sess:
        accounts = pd.read_sql(select(Account), sess.connection())
    return accounts