from utils.database import execute_query
from utils.image_process import preprocess_image, extract_text_from_image, generate_caption_blip


def main():
    select_query = "SELECT id, pub_id, pub_img FROM tweets WHERE pub_img IS NOT NULL AND pub_img_text IS NULL AND pub_img_caption IS NULL LIMIT 100"
    update_query = "UPDATE tweets SET pub_img_text = %s, pub_img_caption = %s WHERE id = %s"

    while True:
        tweets = execute_query(select_query)

        if len(tweets) == 0:
            break

        for tweet in tweets:
            processed_image = preprocess_image(tweet['pub_img'])
            image_text = extract_text_from_image(preprocess_image)
            image_caption = generate_caption_blip(tweet['pub_img'])

            execute_query(update_query, (image_text, image_caption, tweet['id']))



if __name__ == "__main__":
    main()