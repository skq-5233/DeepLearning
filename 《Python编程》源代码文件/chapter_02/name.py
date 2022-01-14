first_name = "ada"
last_name = "lovelace"
full_name = first_name + " " + last_name

message = "Hello, " + full_name.title() + "!"  # title（单词首字母大写-2022-0114）；
print(message)

print('***************欢迎光临**************\n'
      '************************************')
name = 'niuma'
key = '123'

id = 0
while 1:
    use_name = input('请输入用户名：')
    use_key = input('请输入登录密码:')
    if use_name == name and use_key == key:
        print('********登录成功，欢迎进入牛马新世界********\n'
              '********开启牛马之旅**************')
        break
    else:
        if id < 3:
            print('请检查用户名登录密码是否正确，并重新登录！')
            id += 1
        elif id == 3:
            print('今日次数已用完，请明天登录！')
            break
print("Hello Bull_Horse World!!!")

