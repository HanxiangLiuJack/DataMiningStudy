import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
from collections import Counter
import datetime
import json

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 200)

PLOT_COLORS = ["#447394", "#3D9E73", "#C24465", "#E8984C", "#025D87"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)


def basicInfo():
    basic = df.describe(include=['O'])
    print("===========================Dataset basic information===========================")
    print(basic)
    print("======================================End======================================")


def trendingViews():
    fig, ax = plt.subplots()
    _ = sns.distplot(df[df["views"] < 1e6]["views"], kde=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(title="Trending Video Views", xlabel="Views", ylabel="No. of videos")

    plt.show()

    majorView = df[df['views'] < 1e6]['views'].count() / df['views'].count() * 100

    print("%.2f percent of the trending videos have the view count less than 1000000" % majorView)


def nonTrendingViews():
    fig, ax = plt.subplots()
    _ = sns.distplot(dfNon[dfNon["views"] < 1.5e5]["views"], kde=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(title="Non-Trending Video Views", xlabel="Views", ylabel="No. of videos")

    plt.show()

    majorView = dfNon[dfNon['views'] < 1e5]['views'].count() / dfNon['views'].count() * 100
    print("%.2f percent of the non-trending videos have the view count less than 100000" % majorView)


def trendingLikes():
    fig, ax = plt.subplots()
    _ = sns.distplot(df[df["likes"] <= 4e4]["likes"], kde=False,
                     color=PLOT_COLORS[2], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(title="Trending Video Likes", xlabel="Likes", ylabel="No. of videos")
    plt.show()

    majorLikes = df[df['likes'] < 4e4]['likes'].count() / df['likes'].count() * 100
    print("%.2f percent of the trending videos have the like count less than 40000" % majorLikes)


def nonTrendingLikes():
    fig, ax = plt.subplots()
    _ = sns.distplot(dfNon[dfNon["likes"] <= 5e3]["likes"], kde=False,
                     color=PLOT_COLORS[2], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(title="Non-Trending Video Likes",  xlabel="Likes", ylabel="No. of videos")
    plt.show()

    majorLikes = dfNon[dfNon['likes'] < 2.5e3]['likes'].count() / dfNon['likes'].count() * 100
    print("%.2f percent of the non-trending videos have the like count less than 2500" % majorLikes)


def trendingComments():
    fig, ax = plt.subplots()
    _ = sns.distplot(df[df["comment_count"] < 50000]["comment_count"], kde=False, rug=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1},
                     bins=np.linspace(0, 5e4, 49), ax=ax)
    _ = ax.set(title="Trending Video Comments", xlabel="Comment Count", ylabel="No. of videos")
    plt.show()

    majorComments = df[df['comment_count'] < 5000]['comment_count'].count() / df['comment_count'].count() * 100
    print("%.2f percent of the trending videos have the comment count less than 5000" % majorComments)


def nonTrendingComments():
    fig, ax = plt.subplots()
    _ = sns.distplot(dfNon[dfNon["comment_count"] < 10000]["comment_count"], kde=False, rug=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1},
                     bins=np.linspace(0, 1e4, 49), ax=ax)
    _ = ax.set(title="Non-Trending Video Comments",  xlabel="Comment Count", ylabel="No. of videos")
    plt.show()

    majorComments = dfNon[dfNon['comment_count'] < 400]['comment_count'].count() / dfNon['comment_count'].count() * 100
    print("%.2f percent of the non-trending videos have the comment count less than 400" % majorComments)


def containsSymbolWord(s):
    regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
    for w in s.split():
        if regex.search(w) is not None:
            return True
    return False


def titleContainsSymbol():
    df["contains_symbol"] = df["title"].apply(containsSymbolWord)

    value_counts = df["contains_symbol"].value_counts().to_dict()
    fig, ax = plt.subplots()
    _ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'],
               colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
    _ = ax.axis('equal')
    _ = ax.set_title('Title Contains Symbols?')

    plt.show()


def titleLength():
    df["title_length"] = df["title"].apply(lambda x: len(x))

    fig, ax = plt.subplots()
    _ = sns.distplot(df["title_length"], kde=False, rug=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))

    fig, ax = plt.subplots()
    _ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
    _ = ax.set(title="Title length distribution", xlabel="Views", ylabel="Title Length")

    plt.show()


def correlation():
    h_labels = [x.replace('_', ' ').title() for x in
                list(df.select_dtypes(include=['number', 'bool']).columns.values)]

    fig, ax = plt.subplots(figsize=(10, 6))
    _ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels,
                    cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
    _ = ax.set_title('Data correlation')

    plt.show()


def titleFrequentWords():
    title_words = list(df["title"].apply(lambda x: x.split()))
    title_words = [x for y in title_words for x in y]
    counter = Counter(title_words).most_common(50)

    x = []
    y = []

    for index in range(1, len(counter)):
        x.append(counter[index][0])
        y.append(counter[index][1])

    fig, ax = plt.subplots(figsize=(8, 8))
    _ = sns.barplot(x, y, palette=sns.cubehelix_palette(n_colors=100, reverse=True), ax=ax)
    _ = ax.set(title="Frequent words in title", xlabel="Frequent words", ylabel="No. of words")
    plt.xticks(
        rotation=60,
        horizontalalignment='right',
        fontweight='light',
        fontsize='small'
    )

    plt.show()


def tagFrequentWords():
    tag_words = list(df["tags"].apply(lambda x: re.sub(r'[^\w\s]', ' ', x).split(" ")))
    tag_words = [x for y in tag_words for x in y]
    counter = Counter(tag_words).most_common(50)

    x = []
    y = []

    for index in range(1, len(counter)):
        x.append(counter[index][0])
        y.append(counter[index][1])

    fig, ax = plt.subplots(figsize=(10, 8))
    _ = sns.barplot(x, y, palette=sns.cubehelix_palette(n_colors=100, reverse=True), ax=ax)
    _ = ax.set(title="Frequent words in tag", xlabel="Frequent words", ylabel="No. of words")
    plt.xticks(
        rotation=60,
        horizontalalignment='right',
        fontweight='light',
        fontsize='small'
    )

    plt.show()


def descriptionFrequentWords():
    description_words = list(df["description"].apply(lambda x: x.split(" ")))
    description_words = [x for y in description_words for x in y]
    counter = Counter(description_words).most_common(50)

    x = []
    y = []

    for index in range(len(counter)):
        x.append(counter[index][0])
        y.append(counter[index][1])

    fig, ax = plt.subplots(figsize=(10, 8))
    _ = sns.barplot(x, y, palette=sns.cubehelix_palette(n_colors=100, reverse=True), ax=ax)
    _ = ax.set(title="Frequent words in description", xlabel="Frequent words", ylabel="No. of words")
    plt.xticks(
        rotation=60,
        horizontalalignment='right',
        fontweight='light',
        fontsize='small'
    )

    plt.show()


def channelDistribution():
    cdf = df.groupby("channel_title").size().reset_index(name="video_count") \
        .sort_values("video_count", ascending=False).head(60)

    fig, ax = plt.subplots(figsize=(10, 8))
    _ = sns.barplot(x="video_count", y="channel_title", data=cdf,
                    palette=sns.cubehelix_palette(n_colors=60, reverse=True), ax=ax)
    _ = ax.set(title="Channel distribution", xlabel="No. of videos", ylabel="Channel")

    plt.show()


def categoryDistribution():
    with open("./data/CA_category_id.json") as f:
        categories = json.load(f)["items"]
    cat_dict = {}
    for cat in categories:
        cat_dict[int(cat["id"])] = cat["snippet"]["title"]
    df['category_name'] = df['category_id'].map(cat_dict)

    cdf = df["category_name"].value_counts().to_frame().reset_index()
    cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
    fig, ax = plt.subplots()
    _ = sns.barplot(x="category_name", y="No_of_videos", data=cdf,
                    palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    _ = ax.set(title="Category distribution", xlabel="Category", ylabel="No. of videos")
    plt.show()


def publishingDayDistribution():
    df["publishing_day"] = df["publish_time"].apply(
        lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
    df["publishing_hour"] = df["publish_time"].apply(lambda x: x[11:13])
    df.drop(labels='publish_time', axis=1, inplace=True)

    cdf = df["publishing_day"].value_counts() \
        .to_frame().reset_index().rename(columns={"index": "publishing_day", "publishing_day": "No_of_videos"})
    fig, ax = plt.subplots()
    _ = sns.barplot(x="publishing_day", y="No_of_videos", data=cdf,
                    palette=sns.color_palette(['#003f5c', '#374c80', '#7a5195',
                                               '#bc5090', '#ef5675', '#ff764a', '#ffa600'], n_colors=7), ax=ax)
    _ = ax.set(title="Publishing Date distribution", xlabel="Publishing Day", ylabel="No. of videos")

    plt.show()


def publishingTimeDistribution():
    cdf = df["publishing_hour"].value_counts().to_frame().reset_index() \
        .rename(columns={"index": "publishing_hour", "publishing_hour": "No_of_videos"})
    fig, ax = plt.subplots()
    _ = sns.barplot(x="publishing_hour", y="No_of_videos", data=cdf,
                    palette=sns.cubehelix_palette(n_colors=24), ax=ax)
    _ = ax.set(title="Publishing Time distribution", xlabel="Publishing Hour", ylabel="No. of videos")

    plt.show()


df = pd.DataFrame(pd.read_csv('./data/CAvideos.csv', encoding='ISO-8859-1'))
dfNon = pd.DataFrame(pd.read_csv('./data/NonTrendingVideos.csv', encoding='ISO-8859-1'))

# Fill na value
df["description"] = df["description"].fillna(value="")
dfNon["description"] = dfNon["description"].fillna(value="")

basicInfo()
trendingViews()
nonTrendingViews()
trendingLikes()
nonTrendingLikes()
trendingComments()
nonTrendingComments()
titleContainsSymbol()
titleLength()
correlation()
titleFrequentWords()
tagFrequentWords()
descriptionFrequentWords()
channelDistribution()
categoryDistribution()
publishingDayDistribution()
publishingTimeDistribution()
