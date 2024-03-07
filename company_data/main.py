import os
from sqlalchemy import create_engine
import pandas as pd
import database.sql_db as sql_db


def init_sql_db(data_path: str = "./data", mock_data_path: str = "./mock_data", reset: bool = True):
    """
    Initialize the SQL database and load the mock data if available.

    :param data_path:      Data path.
    :param mock_data_path: Mock data path.
    :param reset:          Whether to reset the database.
    """
    # Create the base data path if it doesn't exist:
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Connect to the SQL database:
    sql_connection_url = f"sqlite:///{data_path}/sql.db"
    engine = sql_db.get_engine(sql_connection_url=sql_connection_url)

    # Drop the tables if reset is required:
    if reset:
        sql_db.drop_tables(engine=engine)

    # Create the tables:
    sql_db.create_tables(engine=engine)

    # Check if needed to load mock data:
    if not mock_data_path:
        return

    # Load the mock data:
    products = pd.read_csv(os.path.join(mock_data_path, "products.csv"))
    items = pd.read_csv(os.path.join(mock_data_path, "items.csv"))
    users = pd.read_csv(os.path.join(mock_data_path, "users.csv"))
    stocks = pd.read_csv(os.path.join(mock_data_path, "stocks.csv"))
    purchases = pd.read_csv(os.path.join(mock_data_path, "purchases.csv"))
    stock_to_purchase = pd.read_csv(os.path.join(mock_data_path, "stock_to_purchase.csv"))
    reviews = pd.read_csv(os.path.join(mock_data_path, "reviews.csv"))

    # Insert the mock data into tables:
    products.to_sql(name="product", con=engine, if_exists="replace", index=False)
    items.to_sql(name="item", con=engine, if_exists="replace", index=False)
    users.to_sql(name="user", con=engine, if_exists="replace", index=False)
    stocks.to_sql(name="stock", con=engine, if_exists="replace", index=False)
    purchases.to_sql(name="purchase", con=engine, if_exists="replace", index=False)
    stock_to_purchase.to_sql(name="stock_to_purchase", con=engine, if_exists="replace", index=False)
    reviews.to_sql(name="review", con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    init_sql_db()
    sql_connection_url = f"sqlite:///{'./'}/sql.db"
    engine = sql_db.get_engine(f"sqlite:///data/sql.db")
    items = sql_db.get_items(engine=engine, kinds=["rings", "bracelets"], stones=["no stones"], metals=["white gold"])
    print(items)
    items = sql_db.get_user_items_purchases_history(engine=engine, user_id="6")
    print(items)
