from utils.database import execute_query
from utils.image_process import generate_caption_blip
import sys


def main():
    if len(sys.argv) <= 1:
        print("Nenhum parâmetro foi passado.")
        sys.exit()

    order_by = sys.argv[1]

    if order_by not in ["ASC", "DESC"]:
        print("Parâmetros aceitos: ASC, DESC.")
        sys.exit()

    select_query = f"SELECT id, pub_id, pub_img FROM tweets WHERE pub_img IS NOT NULL AND pub_img_caption IS NULL ORDER BY {order_by} LIMIT 100"
    update_query = "UPDATE tweets SET pub_img_caption = %s WHERE id = %s"

    while True:
        tweets = execute_query(select_query)

        if len(tweets) == 0: break

        for tweet in tweets:
            try:
                image_caption = generate_caption_blip(tweet['pub_img'])
                execute_query(update_query, (image_caption, tweet['id']))
            except Exception as e:
                execute_query(update_query, ("No image", tweet['id']))



if __name__ == "__main__":
    main()