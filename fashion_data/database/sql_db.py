"""
SQL Database for the jewelry agent demo.
"""
import datetime
import os
from typing import List, Optional, Tuple, Literal

import pandas as pd
from sqlalchemy import (
    ForeignKey,
    Boolean,
    Date,
    Float,
    Integer,
    String,
    bindparam,
    create_engine,
    insert,
    select,
    update,
    Column,
    Table,
    Engine,
    func,
    or_,
)
from sqlalchemy.orm import (
    relationship,
    Mapped,
    declarative_base,
    mapped_column,
    sessionmaker,
)


ID_LENGTH = 32

KindType = Literal[
    "flip-flop",
    "jacket",
    "long pants",
    "polo shirt",
    "t-shirt",
    "shirt",
    "shoes",
    "short pants",
    "sweater",
    "sweatshirt",
    "swimsuit",
    "tanktop",
]
SortByType = Literal["highest_price", "lowest_price", "newest"]

Base = declarative_base()


class Item(Base):
    """
    Item of clothing in the store.

    :arg item_id:                   The unique identifier of the item.
    :arg date_added:                The date that this item was added to the store.
    :arg name:                      The name of the item.
    :arg kind:                      The kind of the item, for example: t-shirt, long pants, etc.
    :arg description:               A description of the item.
    :arg fit:                       The fit of the item, for example: slim, regular, etc.
    :arg colors:                    The colors of the item.
    :arg material:                  The material of the item.
    :arg styles:                    The styles of the item.
    :arg price:                     The price of the item.
    :arg image:                     The image file name to the item.
    """
    __tablename__ = "item"

    # Columns:
    item_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    date_added: Mapped[datetime.date] = mapped_column(Date())
    name: Mapped[str] = mapped_column(String(length=15))
    kind: Mapped[str] = mapped_column(String(length=15))
    description: Mapped[str] = mapped_column(String(length=1000))
    fit: Mapped[str] = mapped_column(String(length=15))
    colors: Mapped[str] = mapped_column(String(length=100))
    material: Mapped[str] = mapped_column(String(length=100))
    styles: Mapped[str] = mapped_column(String(length=100))
    price: Mapped[float] = mapped_column(Float(precision=2))
    image: Mapped[str] = mapped_column(String(length=1000))


def get_engine(sql_connection_url: str) -> Engine:
    """
    Get the SQL database engine.

    :param sql_connection_url: The SQL connection URL.

    :return:                   The SQL database engine.
    """
    return create_engine(sql_connection_url)


def create_tables(engine: Engine):
    """
    Create the database tables.
    """
    Base.metadata.create_all(engine)


def drop_tables(engine: Engine):
    """
    Drop the database tables.
    """
    # Delete the schema's tables:
    Base.metadata.drop_all(engine)


def get_items(
    engine: Engine,
    ids: List[str] = None,
    kind: List[KindType] = None,
    color: List[str] = None,
    fit: List[str] = None,
    style: List[str] = None,
    materials: List[str] = None,
    min_price: float = None,
    price: float = None,
    sort_by: SortByType = "lowest_price",
) -> pd.DataFrame:
    """
    Get the items from the database.

    :param engine:      A SQL database engine.
    :param ids:         The unique identifiers of the items.
    :param kind:       Kinds of products to filter by.
    :param color:      Colors of items to filter by.
    :param fit:        Fits of items to filter by.
    :param style:      Styles of items to filter by.
    :param materials:   Materials of items to filter by.
    :param min_price:   The minimum price of the items.
    :param price:   The maximum price of the items.
    :param sort_by:     Sort the items by one of the following: highest_price, lowest_price, newest.

    :return: A DataFrame of the items.
    """
    def or_like_criteria(column, values):
        or_criteria = [column.in_(values)]
        for value in values:
            or_criteria.append(column.like(f"%{value}%"))
        return or_(*or_criteria)

    with engine.connect() as conn:
        query = select(Item)

        if min_price:
            query = query.where(Item.price >= min_price)
        if price:
            query = query.where(Item.price <= price)
        if ids:
            query = query.where(Item.item_id.in_(ids))
        if kind:
            query = query.where(Item.kind.in_(kind))
        if color:
            query = query.where(or_like_criteria(column=Item.colors, values=color))
        if fit:
            query = query.where(or_like_criteria(column=Item.fit, values=fit))
        if style:
            query = query.where(or_like_criteria(column=Item.styles, values=style))
        if materials:
            query = query.where(or_like_criteria(column=Item.material, values=materials))

        if sort_by:
            if sort_by == "highest_price":
                query = query.order_by(Item.price.desc())
            elif sort_by == "lowest_price":
                query = query.order_by(Item.price.asc())
            elif sort_by == "newest":
                query = query.order_by(Item.date_added.desc())

        items = conn.execute(query).all()

    return pd.DataFrame(items)
