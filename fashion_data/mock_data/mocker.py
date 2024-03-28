import datetime
import os
import random
import uuid

import pandas as pd


def _generate_items(
    catalog_items: list[dict],
    min_date_added: str,
    max_date_added: str,
    price_delta: float = 0.25,
) -> list[dict]:
    kind_to_price_map = {
        'flip-flop': 30,
        'jacket': 150,
        'long pants': 60,
        'polo shirt': 40,
        't-shirt': 20,
        'shirt': 80,
        'shoes': 120,
        'short pants': 40,
        'sweater': 80,
        'sweatshirt': 50,
        'swimsuit': 60,
        'tanktop': 20,
    }
    min_date_added = datetime.datetime.strptime(min_date_added, "%m/%d/%Y")
    max_date_added = datetime.datetime.strptime(max_date_added, "%m/%d/%Y")

    for item in catalog_items:
        item_id = str(uuid.uuid4()).replace("-", "")
        date_added = min_date_added + datetime.timedelta(
            days=random.randint(0, (max_date_added - min_date_added).days)
        )
        name = item["filename"].split(".")[0].replace("_", " ").title()
        price = kind_to_price_map[item["kind"]]
        price = (random.randint(price, int(price * (1+price_delta))) // 10) * 10
        item.update(
            item_id=item_id,
            date_added=date_added,
            name=name,
            price=price,
        )

    return catalog_items


def generate_mock_data(
    sources_directory: str = "./sources", output_directory: str = "./"
):
    items = pd.read_csv(os.path.join(sources_directory, "catalog.csv")).to_dict(orient="records")
    items = _generate_items(
        catalog_items=items,
        min_date_added="01/01/2024",
        max_date_added="03/21/2024",
    )
    items = pd.DataFrame(items)
    items.rename(columns={"filename": "image", "caption": "description", "style": "styles"}, inplace=True)
    items.to_csv(os.path.join(output_directory, "items.csv"), index=False)


if __name__ == "__main__":
    generate_mock_data()
