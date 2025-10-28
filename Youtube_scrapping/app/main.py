from googleapiclient.discovery import build
import csv
import time


api_key = "AIzaSyCjqVNXd8yME5YhoORvHZFPQtWzd3Q1uzE"
video_id = "pJlnxbO5N2g" #1) Zi_XLOBDo_Y, 2) 6CYr0JQv1oQ, 3) pJlnxbO5N2g - 3 viral videos comments was taken

youtube = build('youtube', 'v3', developerKey=api_key)


csv_file = "youtube_comments_last.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["comment_id", "author", "text", "published_at", "like_count", "reply_count", "video_id"])


    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    total_comments = 0
    while request:
        response = request.execute()

        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comment_id = item['snippet']['topLevelComment']['id']
            author = snippet['authorDisplayName']
            text = snippet['textDisplay']
            published_at = snippet['publishedAt']
            like_count = snippet['likeCount']
            reply_count = item['snippet']['totalReplyCount']

            writer.writerow([comment_id, author, text, published_at, like_count, reply_count, video_id])
            total_comments += 1

        print(f"✅ {total_comments} comments downloaded...")


        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100
            )
        else:
            break


        time.sleep(0.1)

print(f"🎉 All comments saved to '{csv_file}'. Total comments: {total_comments}")
