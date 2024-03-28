import os
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
    items = pd.read_csv(os.path.join(mock_data_path, "items.csv"))

    # Insert the mock data into tables:
    items.to_sql(name="item", con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    init_sql_db()
    sql_connection_url = f"sqlite:///{'./'}/sql.db"
    engine = sql_db.get_engine(f"sqlite:///data/sql.db")
    items = sql_db.get_items(engine=engine, kind=["t-shirt", "sweatshirt", "sweater"], color=["red", "black", "blue"], min_price=25, sort_by="highest_price")
    print(items)
