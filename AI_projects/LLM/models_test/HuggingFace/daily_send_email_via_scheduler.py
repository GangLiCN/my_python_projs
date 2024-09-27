
import smtplib
import schedule
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email():
    # 邮件服务器配置
    smtp_server = 'smtp.163.com'  # 替换为你的 SMTP 服务器
    smtp_port = 465  # 使用 SSL 的 SMTP 端口通常是 465
    smtp_username = 'krislee@163.com'  # 替换为你的邮箱地址
    smtp_password = 'Beauti2018__'  # 替换为你的邮箱密码

    # 邮件内容
    from_email = smtp_username
    to_email = 'krislee@163.com'  # 替换为接收者的邮箱地址
    subject = '每日自动发送邮件测试'
    body = '这是一封测试邮件，如果你能正确阅读，证明定时邮件发送程序工作正常!'

    # 创建邮件
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # 发送邮件
    try:
        # 使用 SSL 连接到 SMTP 服务器
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print('邮件发送成功')
    except Exception as e:
        print(f'邮件发送失败: {e}')

    # 取消任务
    return schedule.CancelJob


# 当前时间
now = datetime.now()
target_time = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)  # 计划任务时间为下午3点

# 检查当前时间是否已经过了计划任务时间
if now < target_time:
    schedule.every().day.at("15:00").do(send_email)  # 每天下午3点运行任务

    # 保持脚本运行直到任务执行
    while True:
        pending = schedule.run_pending()
        if pending == schedule.CancelJob:
            break
        time.sleep(1)
else:
    print("当前时间已超过任务执行时间，任务今天将不再执行。")
