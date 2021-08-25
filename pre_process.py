import datetime
from datetime import *
import pandas as pd
import numpy as np


# 完整性验证
def user_action_check():
    df_user = pd.read_csv('data/user_info.csv')
    df_sku = df_user.loc[:, 'user_id'].to_frame()
    df_login = pd.read_csv('data/login_day.csv')
    print('Is action of login. from User file? ', len(df_login) == len(pd.merge(df_sku, df_login)))
    df_visit = pd.read_csv('data/visit_info.csv')
    print('Is action of visit. from User file? ', len(df_visit) == len(pd.merge(df_sku, df_visit)))
    df_result = pd.read_csv('data/result.csv')
    print('Is action of result. from User file? ', len(df_result) == len(pd.merge(df_sku, df_result)))


# 查看是否有重复记录
def deduplicate(filepath, filename, newpath):
    df_file = pd.read_csv(filepath, encoding='utf-8')
    before = df_file.shape[0]
    df_file.drop_duplicates(inplace=True)
    after = df_file.shape[0]
    n_dup = before-after
    print ('No. of duplicate records for ' + filename + ' is: ' + str(n_dup))
    if n_dup != 0:
        df_file.to_csv(newpath, index=None)
    else:
        print('no duplicate records in ' + filename)


# 展示表的信息
def table_information():
    df_user = pd.read_csv('data/user_info.csv')
    df_user.head()
    print(df_user.shape)
    df_user.info()

    df_login = pd.read_csv('data/login_day.csv')
    df_login.head()
    print(df_login.shape)
    df_login.info()

    df_visit = pd.read_csv('data/visit_info.csv')
    df_visit.head()
    print(df_visit.shape)
    df_visit.info()

    print('visit_info表中click_buy值得分布情况：')
    print(df_visit['click_buy'].value_counts())


def modify_platform(num):
    if num == 9.2969:
        return 0
    if num == 13.557:
        return 1


def modify_time(time):
    temp = time.split(" ")[0]
    temp = pd.to_datetime(temp)
    return temp


# 删除一些无用字段(手机型号,空城市)
def process_user():
    df_user = pd.read_csv('data/user_all_info.csv')
    df_user.drop(columns=['model_num', 'app_num'], inplace=True)
    df_user.dropna(axis=0, subset=['city_num'])
    df_user.query("city_num!='error'", inplace=True)
    df_user['user_id'] = df_user['user_id']-2000000000000000
    df_user['platform_num'] = df_user['platform_num'].apply(modify_platform)
    # df_user['first_order_time'] = df_user['first_order_time'].apply(modify_time)
    # df_user['year'] = df_user['first_order_time'].apply(lambda x: x.year)
    # df_user['month'] = df_user['first_order_time'].apply(lambda x: x.month)
    # df_user['weekday'] = df_user['first_order_time'].apply(lambda x: x.weekday()+1)
    df_user.to_csv('data/user_all_info.csv')


def process_login():
    df_login = pd.read_csv('data/user_all_info.csv')
    df_login = df_login[df_login['login_day'] >=0]
    df_login = df_login[~(df_login['add_friend'] + df_login['add_group'] == 0)]
    df_login['subscribe_num'] = df_login['chinese_subscribe_num'] + df_login['math_subscribe_num']
    df_login.drop(columns=['add_friend', 'add_group', 'chinese_subscribe_num', 'math_subscribe_num'], inplace=True)
    df_login.to_csv('data/user_all_info.csv')


def process_visit():
    df_visit = pd.read_csv('data/user_all_info.csv')
    df_visit['main_home'] = df_visit['main_home'] + df_visit['main_home2']
    df_visit['main_mime'] = df_visit['mainpage'] + df_visit['schoolreportpage'] + \
                            df_visit['main_mime'] + df_visit['lightcoursetab']
    df_visit['place_count'] = df_visit['main_learnpark'] + df_visit['partnergamebarrierspage']+ \
                              df_visit['evaulationcenter']
    df_visit['next_count'] = df_visit['progress_bar']+df_visit['ppt']+df_visit['task']+df_visit['video_play'] + \
                             df_visit['video_read']+df_visit['next_nize']+df_visit['answer_task']
    df_visit['chapter_course_count'] = df_visit['chapter_module']+df_visit['course_tab']+df_visit['slide_subscribe']
    df_visit['click_dialog'] = df_visit['click_dialog']+ df_visit['click_notunlocked']
    df_visit.drop(columns=['click_buy', 'main_home2', 'mainpage', 'schoolreportpage'], inplace=True)
    df_visit.drop(columns=['lightcoursetab', 'main_learnpark', 'partnergamebarrierspage'], inplace=True)
    df_visit.drop(columns=['evaulationcenter', 'progress_bar', 'ppt', 'task', 'video_play'], inplace=True)
    df_visit.drop(columns=['video_read', 'next_nize', 'answer_task'], inplace=True)
    df_visit.drop(columns=['chapter_module', 'course_tab', 'slide_subscribe'], inplace=True)
    df_visit.drop(columns=['click_notunlocked'], inplace=True)
    df_visit.to_csv('data/user_all_info.csv')


def merge_table():
    df_user = pd.read_csv('data/user_info.csv')
    df_login = pd.read_csv('data/login_day.csv')
    df_visit = pd.read_csv('data/visit_info.csv')
    df_result = pd.read_csv('data/result.csv')
    sum_table = pd.merge(df_user, df_login, on='user_id', how='outer')
    sum_table = pd.merge(sum_table, df_visit, on='user_id', how='outer')
    sum_table = pd.merge(sum_table, df_result, on='user_id', how='outer')
    sum_table['result'].fillna(0, inplace=True)
    sum_table.to_csv('data/user_all_info.csv')


if __name__ == '__main__':
    # deduplicate('data/login_day.csv', 'login_day', 'data/login_day_dup.csv')
    # deduplicate('data/user_info.csv', 'user_info', 'data/user_info_dup.csv')
    # deduplicate('data/visit_info.csv', 'visit_info', 'data/visit_info_dup.csv')

    # user_action_check()

    # merge_table()

    # process_user()
    # process_login()
    # process_visit()

    all_info = pd.read_csv('data/user_all_info.csv')
    all_info.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], inplace=True)
    # all_info.drop(columns=['Unnamed: 0'], inplace=True)
    all_info.to_csv('data/user_all_info.csv')










