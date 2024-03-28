# Copyright 2024 Iguazio
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

Base = declarative_base()

#: Association table between stocks and purchases for many-to-many relationship.
stock_to_purchase = Table(
    "stock_to_purchase",
    Base.metadata,
    Column(
        "stock_id",
        String(length=ID_LENGTH),
        ForeignKey("stock.stock_id"),
        primary_key=True,
    ),
    Column(
        "purchase_id",
        String(length=ID_LENGTH),
        ForeignKey("purchase.purchase_id"),
        primary_key=True,
    ),
)


class Product(Base):
    """
    A product is the main entity in the jewelry database. It represents a piece of jewelry in all its variations.

    :arg product_id:  The unique identifier of the product.
    :arg name:        The name of the product.
    :arg kind:        The kind of the product, one of: rings, bracelets, necklaces, earrings.
    :arg collections: The collections that this product belongs to.
    :arg gifts:       The gifts that this product is recommended as.
    :arg items:       The items that belong to this product. Usually separated by metals and stones.
    """
    __tablename__ = "product"

    # Columns:
    product_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    name: Mapped[str] = mapped_column(String(length=30))
    kind: Mapped[str] = mapped_column(String(length=30))
    collections: Mapped[str] = mapped_column(String(length=30))
    gifts: Mapped[str] = mapped_column(String(length=30))

    # Relationships:
    items: Mapped[List["Item"]] = relationship(  # one-to-many
        back_populates="product", lazy=True
    )


class Item(Base):
    """
    An item is a specific variation of a product that holds all of its properties.

    :arg item_id:                   The unique identifier of the item.
    :arg product_id:                The unique identifier of the product that this item belongs to.
    :arg date_added:                The date that this item was added to the store.
    :arg description:               A description of the item.
    :arg colors:                    The colors of the item.
    :arg metals:                    The metals of the item. Can be one or more of the following metals: yellow gold,
                                    white gold, pink gold, silver and platinum.
    :arg stones:                    The material of the item. Default is None (“no stones”).
    :arg stones_carat_total_weight: The total weight of the stones in the item.
    :arg image:                     The URL to the item's image.
    :arg product:                   The product that this item belongs to.
    :arg stocks:                    The stocks of this item.
    :arg reviews:                   The reviews of this item.
    """
    __tablename__ = "item"

    # Columns:
    item_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("product.product_id")
    )
    date_added: Mapped[datetime.date] = mapped_column(Date())
    description: Mapped[str] = mapped_column(String(length=1000))
    colors: Mapped[str] = mapped_column(String(length=100))
    metals: Mapped[str] = mapped_column(String(length=100))
    stones: Mapped[str] = mapped_column(String(length=100), default="no stones", nullable=True)
    stones_carat_total_weight: Mapped[float] = mapped_column(Float(precision=2), nullable=True)
    image: Mapped[str] = mapped_column(String(length=1000))

    # Relationships:
    product: Mapped["Product"] = relationship(  # one-to-many
        back_populates="items", lazy=True
    )
    stocks: Mapped[List["Stock"]] = relationship(  # one-to-many
        back_populates="item", lazy=True
    )
    reviews: Mapped[List["Review"]] = relationship(  # one-to-many
        back_populates="item", lazy=True
    )


class Stock(Base):
    """
    A stock is a specific item that is available for purchase in a specific size and price.

    :arg stock_id:  The unique identifier of the stock.
    :arg item_id:   The unique identifier of the item that this stock belongs to.
    :arg size:      The size of the item (for example, a diameter for rings: 4).
    :arg amount:    The amount of the item in the specific size in stock.
    :arg price:     The price of the item in dollars.
    :arg item:      The item that this stock belongs to.
    :arg purchases: The purchases that this stock is part of.
    """
    __tablename__ = "stock"

    # Columns:
    stock_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    item_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("item.item_id")
    )
    size: Mapped[float] = mapped_column(Float(precision=1))
    amount: Mapped[int] = mapped_column(Integer())
    price: Mapped[float] = mapped_column(Float(precision=4))

    # Relationships:
    item: Mapped["Item"] = relationship(back_populates="stocks", lazy=True)  # 1-to-many
    purchases: Mapped[List["Purchase"]] = relationship(  # many-to-many
        secondary=stock_to_purchase,
        back_populates="stocks",
        lazy=True,
    )


class User(Base):
    """
    A user is a person that uses the store's website.

    :arg user_id:       The unique identifier of the user.
    :arg first_name:    The first name of the user.
    :arg last_name:     The last name of the user.
    :arg date_of_birth: The date of birth of the user.
    :arg email:         The email of the user.
    :arg phone:         The phone number of the user.
    :arg address:       The address of the user.
    :arg purchases:     The purchases that the user made.
    :arg reviews:       The reviews that the user wrote.
    """
    __tablename__ = "user"

    # Columns:
    user_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(length=30))
    last_name: Mapped[str] = mapped_column(String(length=30))
    date_of_birth: Mapped[datetime.date] = mapped_column(Date())
    email: Mapped[str] = mapped_column(String(length=30), nullable=True)
    phone: Mapped[str] = mapped_column(String(length=30), nullable=True)
    address: Mapped[str] = mapped_column(String(length=100), nullable=True)

    # Relationships:
    purchases: Mapped[List["Purchase"]] = relationship(  # 1-to-many
        back_populates="user", lazy=True
    )
    reviews: Mapped[List["Review"]] = relationship(  # 1-to-many
        back_populates="user", lazy=True
    )


class Purchase(Base):
    """
    A purchase is a user's purchase of a specific stock.

    :arg purchase_id: The unique identifier of the purchase.
    :arg user_id:     The unique identifier of the user that made the purchase.
    :arg date:        The date of the purchase.
    :arg user:        The user that made the purchase.
    :arg stocks:      The stocks items that were purchased.
    """
    __tablename__ = "purchase"

    # Columns:
    purchase_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("user.user_id")
    )
    date: Mapped[datetime.date] = mapped_column(Date())

    # Relationships:
    user: Mapped["User"] = relationship(  # 1-to-many
        back_populates="purchases", lazy=True
    )
    stocks: Mapped[List["Stock"]] = relationship(  # many-to-many
        secondary=stock_to_purchase, back_populates="purchases", lazy=True
    )


class Review(Base):
    """
    A review is a user's opinion about a specific item.

    :arg review_id:    The unique identifier of the review.
    :arg item_id:      The unique identifier of the item that this review belongs to.
    :arg user_id:      The unique identifier of the user that wrote the review.
    :arg date:         The date that the review was written.
    :arg text:         The text of the review.
    :arg rating:       The rating of the review. An integer between 1 and 5.
    :arg is_recommend: Whether the user recommends the item or not.
    :arg item:         The item that this review belongs to.
    :arg user:         The user that wrote the review.
    """
    __tablename__ = "review"

    # Columns:
    review_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    item_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("item.item_id")
    )
    user_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("user.user_id")
    )
    date: Mapped[datetime.date] = mapped_column(Date())
    text: Mapped[str] = mapped_column(String(length=1000), default="", nullable=True)
    rating: Mapped[int] = mapped_column(Integer())
    is_recommend: Mapped[bool] = mapped_column(Boolean())

    # Relationships:
    item: Mapped["Item"] = relationship(  # 1-to-many
        back_populates="reviews", lazy=True
    )
    user: Mapped["User"] = relationship(  # 1-to-many
        back_populates="reviews", lazy=True
    )


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
        kinds: List[Literal["rings", "necklaces", "bracelets", "earrings"]] = None,
        colors: List[str] = None,
        metals: List[str] = None,
        stones: List[str] = None,
        collections: List[str] = None,
        gifts: List[str] = None,
        min_price: float = None,
        max_price: float = None,
        sort_by: Literal["highest_price", "lowest_price", "best_reviews", "best_seller", "newest"] = None,
) -> pd.DataFrame:
    """
    Get the items from the database.

    :param engine:      A SQL database engine.
    :param kinds:       Kinds of products to filter by.
    :param colors:      Colors of items to filter by.
    :param metals:      Metals of items to filter by.
    :param stones:      Stones of items to filter by.
    :param collections: Collections of products to filter by.
    :param gifts:       Gifts of products to filter by.
    :param min_price:   The minimum price of the items.
    :param max_price:   The maximum price of the items.
    :param sort_by:     Sort the items by one of the following: highest_price, lowest_price, best_reviews, best_seller,
                        newest.

    :return: A DataFrame of the items.
    """
    with engine.connect() as conn:
        items_total_purchases_query = (
            select(
                Stock.item_id,
                func.count(stock_to_purchase.c.stock_id).label("total_purchases")
            ).join(
                stock_to_purchase,
                Stock.stock_id == stock_to_purchase.c.stock_id,
                isouter=True,
            ).group_by(Stock.item_id).subquery()
        )
        items_average_reviews_query = (
            select(Review.item_id, func.avg(Review.rating).label("average_rating")).
            group_by(Review.item_id).subquery()
        )
        items_average_price_query = (
            select(Stock.item_id, func.avg(Stock.price).label("price")).
            group_by(Stock.item_id).subquery()
        )

        query = (
            select(Product.name, Product.kind, Item, items_average_price_query.c.price).
            join(Item.product).
            join(items_average_price_query).
            join(items_total_purchases_query, isouter=True).
            join(items_average_reviews_query, isouter=True).
            group_by(Item.item_id)
        )

        if min_price:
            query = query.where(items_average_price_query.c.price >= min_price)
        if max_price:
            query = query.where(items_average_price_query.c.price <= max_price)

        if kinds:
            query = query.where(Product.kind.in_(kinds))
        if colors:
            or_criteria = [Item.colors.in_(colors)]
            for color in colors:
                or_criteria.append(Item.colors.like(f"%{color}%"))
            query = query.where(or_(*or_criteria))
        if metals:
            or_criteria = [Item.metals.in_(metals)]
            for metal in metals:
                or_criteria.append(Item.metals.like(f"%{metal}%"))
            query = query.where(or_(*or_criteria))
        if stones:
            or_criteria = [Item.stones.in_(stones)]
            for stone in stones:
                or_criteria.append(Item.stones.like(f"%{stone}%"))
            query = query.where(or_(*or_criteria))
        if collections:
            or_criteria = [Product.collections.in_(collections)]
            for collection in collections:
                or_criteria.append(Product.collections.like(f"%{collection}%"))
            query = query.where(or_(*or_criteria))
        if gifts:
            or_criteria = [Product.gifts.in_(gifts)]
            for gift in gifts:
                or_criteria.append(Product.gifts.like(f"%{gift}%"))
            query = query.where(or_(*or_criteria))

        if sort_by:
            if sort_by == "highest_price":
                query = query.order_by(Item.stones_carat_total_weight.desc())
            elif sort_by == "lowest_price":
                query = query.order_by(Item.stones_carat_total_weight.asc())
            elif sort_by == "best_reviews":
                query = query.order_by(items_average_reviews_query.c.average_rating.desc())
            elif sort_by == "best_seller":
                query = query.order_by(items_total_purchases_query.c.total_purchases.desc())
            elif sort_by == "newest":
                query = query.order_by(Item.date_added.desc())

        items = conn.execute(query).all()

    return pd.DataFrame(items)


def get_user_items_purchases_history(
        engine: Engine, user_id: str, last_n_purchases: int = 5
) -> pd.DataFrame:
    """
    Get the user's items purchase history.

    :param engine:           A SQL database engine.
    :param user_id:          The user's unique identifier.
    :param last_n_purchases: The last n purchases to return.

    :return: A DataFrame of the user's items purchase history.
    """
    with engine.connect() as conn:
        query = (
            select(Item).join(Stock).join(stock_to_purchase).join(Purchase).
            where(Purchase.user_id == user_id).
            order_by(Purchase.date.desc()).distinct()
        )
        if last_n_purchases:
            query = query.limit(last_n_purchases)
        purchases = conn.execute(query).all()

    return pd.DataFrame(purchases)


def get_item_stocks(engine: Engine, item_id: str = None, product_name: str = None, metal: str = None) -> pd.DataFrame:
    """
    Get the item's stocks. The item can be retrieved by its unique identifier or by its product name and metal.

    :param engine:       A SQL database engine.
    :param item_id:      The item's unique identifier.
    :param product_name: The product's name.
    :param metal:        The metal of the item.

    :return: A DataFrame of the item's stocks.
    """
    with engine.connect() as conn:
        if item_id:
            query = select(Stock).where(Stock.item_id == item_id)
        else:
            assert product_name and metal, "product_name and metal must be provided together"

            query = (
                select(Stock).join(Item).join(Product).
                where(Product.name == product_name).
                where(Item.metals.like(f"%{metal}%"))
            )
        stocks = conn.execute(query).all()

    return pd.DataFrame(stocks)
