import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from src.chains.base import ChainRunner
from src.data.bank_db import (create_tables, insert_accounts, update_accounts, get_accounts)

@tool
def get_inventory() -> pd.DataFrame:
    """
    A tool to get all current store inventory.
    """
    stock = (
        "Cotton T-shirt, Crew neck short sleeve white cotton tee, In stock, S-XL, Loose fit, Plain, $12"
        "Flannel Button-Up, Red and black plaid long sleeve flannel shirt, In stock, M-XXL, Regular fit, Plaid, $25"
        "Blue Jeans, Dark wash classic 5-pocket skinny denim jeans, Low stock, 28-36 waist, Skinny fit, Plain, $40"
        "Yoga Pants, Solid black stretchy fitted yoga pants, In stock, XS-L, Skinny fit, Plain, $35"
        "Hooded Sweatshirt, Gray pullover hoodie with kangaroo pocket, In stock, S-XXL, Loose fit, Plain, $28"
        "Leather Jacket, Black faux leather moto jacket with asymmetrical zip, Out of stock, Waitlist only, Fitted,"
        " Plain, $99"
        "Maxi Dress, Red floral print knee length maxi dress, In stock, 0-16, Loose fit, Floral pattern, $49"
        "Pleated Skirt, Navy A-line knee length polyester skirt, Low stock, 4-14, A-line fit, Plain, $29"
        "Slip Dress, Black satin cami slip midi dress, In stock, S-L, Fitted, Plain, $59"
        "Wool Peacoat, Charcoal double breasted wool blend peacoat, In stock, 36-48, Regular fit, Plain, $159"
    )
    return stock



class Agent(ChainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = ChatOpenAI(model="gpt-4")

    def _get_agent(self, event):
        tools = [get_inventory]
        llm_with_tools = self.llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    fashion_prompt,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                        x["intermediate_steps"]
                    ),
                }
                | prompt
                | llm_with_tools
                | OpenAIToolsAgentOutputParser()
        )
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def _run(self, event):
        agent = self._get_agent(event)
        response = list(agent.stream({"input": event.query}))
        return {"answer": response[-1]["messages"][-1].content, "sources": ""}



if __name__ == "__main__":
    agent = Agent()


fashion_trends = (
    "Balance lengths: Pair long tops with short bottoms and vice versa to create visual balance and avoid overwhelming your silhouette."
    "Color coordination: Coordinate colors in your outfit by choosing complementary or analogous hues. Consider the color wheel to create harmonious combinations."
    "Fit harmony: Ensure that your top and bottom fit well together. For example, if you're wearing a fitted top, opt for slim or straight-leg bottoms to maintain a balanced silhouette."
    "Layering textures: Combine different textures in your outfit to add depth and interest. For instance, pair a chunky knit sweater with sleek leather leggings for a contrasting yet cohesive look."
    "Proportional dressing: Maintain balanced proportions by pairing loose-fitting garments with more tailored pieces. This can prevent your outfit from looking too baggy or shapeless."
    "Color blocking: Experiment with color blocking by wearing solid blocks of color in your outfit. Keep the color combinations bold yet harmonious for a striking look."
    "Play with patterns: Mix patterns carefully by choosing pieces with complementary scales or motifs. Consider pairing a small-scale print with a larger one to create visual interest without overwhelming the eye."
    "Contrast wisely: Incorporate contrast into your outfit through color, texture, or silhouette. For instance, pair a structured blazer with flowy pants for a balanced contrast of tailoring and fluidity."
    "Lengthen with vertical lines: Opt for vertical stripes or elongating details like vertical seams to create the illusion of a longer, leaner silhouette."
    "Harmonize shoe height: Choose shoes that harmonize with the length of your pants or skirt. For example, wear heels with wide-leg trousers to elongate your legs or opt for flats with cropped pants for a modern, casual look."
    "Consider neckline and hemline: Pay attention to the neckline of your top and the hemline of your bottom to ensure they complement each other. For instance, pair a V-neck blouse with a midi skirt for a flattering balance of proportions."
    "Accentuate the waist: Define your waist by pairing tops and bottoms that emphasize this area. Consider tucking in tops or adding a belt to create a more defined silhouette."
    "Experiment with layers: Layering different clothing pieces can add depth and dimension to your outfit. Play with lengths, textures, and colors to create visually dynamic combinations."
    "Monochrome magic: Embrace monochromatic outfits by wearing pieces in varying shades of the same color. This creates a cohesive and elongating effect, perfect for a sleek and sophisticated look."
    "Patterns:"
    "Stripes and Florals: Pairing a striped top with a floral skirt or vice versa can create a playful and visually interesting contrast."
    "Plaid and Denim: Combining a plaid shirt with denim jeans or a denim skirt can achieve a classic and casual Americana-inspired look."
    "Polka Dots and Stripes: Mixing polka dots with stripes adds a whimsical and dynamic touch to your outfit."
    "Animal Print and Neutrals: Incorporating animal print, such as leopard or snakeskin, with neutral colors like black, white, or beige creates a chic and sophisticated ensemble."
    "Geometric Prints and Solids: Pairing geometric prints, such as triangles or chevrons, with solid colors allows the pattern to stand out while maintaining balance in the outfit."
    "Tropical Prints and Brights: Mixing tropical prints with bold, vibrant colors like hot pink, turquoise, or sunny yellow can evoke a fun and tropical vibe."
    "Gingham and Florals: Combining gingham with floral patterns creates a charming and feminine look, perfect for spring or summer."
    "Houndstooth and Leather: Adding a touch of edge to your outfit, pairing houndstooth with leather accents creates a stylish and modern ensemble."
    "Tartan and Knits: Mixing tartan prints with cozy knits adds warmth and texture to your outfit, perfect for autumn and winter styling."
    "Abstract Prints and Pastels: Pairing abstract prints with soft pastel colors creates a contemporary and artsy look with a subtle touch of femininity."
    "Colors:"
    "Navy Blue and Mustard Yellow: This combination offers a sophisticated yet vibrant contrast that works well in both casual and formal outfits."
    "Burgundy and Blush Pink: Pairing rich burgundy with delicate blush pink creates a romantic and elegant color palette, perfect for special occasions."
    "Emerald Green and Gold: The luxurious combination of emerald green and gold exudes opulence and sophistication, ideal for eveningwear or festive occasions."
    "Teal and Coral: Combining the cool tones of teal with the warm hues of coral creates a striking and harmonious contrast that's perfect for summer outfits."
    "Gray and Lavender: This soft and understated combination of gray and lavender creates a calming and sophisticated look, perfect for everyday wear."
    "Black and White: A timeless and classic pairing, black and white create a chic and versatile color palette suitable for any occasion."
    "Olive Green and Rust: Mixing earthy tones like olive green with warm, rusty hues creates a cozy and autumnal color palette perfect for fall fashion."
    "Navy Blue and Camel: Combining the richness of navy blue with the warmth of camel creates a timeless and refined color palette suitable for both casual and formal attire."
    "Turquoise and Coral: The vibrant combination of turquoise and coral creates a lively and energetic color palette, perfect for summer outfits and beachwear."
    "Taupe and Cream: This subtle and sophisticated combination of taupe and cream creates a soft and elegant color palette that's perfect for creating understated chic looks."

)

fashion_prompt = (
    "You are a fashion assistant, you can help users with fashion advice and recommendations."
    "Your job is to help users buy cloths and accessories that are in fashion, with the tools you have."
    "remember to check for sizes and availability of the items, and then fashion conpatibility."
    "When giving advice, make sure to consider the current fashion trends bellow: "
    f"{fashion_trends}"
    "When talking to the user keep your answers short and to the point, be body positive and inclusive."
    "Encourage the customer and praise their choices, and always be polite and respectful."
    "Always try to give the customer more than one option, give them the option to choose."
)

