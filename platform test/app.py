import streamlit as st


def main():
    st.title("ESG评分演示")

    # 用户输入文本
    user_input = st.text_area("输入公司的ESG相关文本进行评估", "在这里输入文本...")

    # 按钮触发模型调用
    if st.button("获取评分"):
        # 这里调用您的GPT模型API
        # 假设API响应中包含 'score' 字段表示ESG评分
        # response = some_api_call(user_input)
        # score = response['score']
        # 模拟得分
        score = 85  # 假设这是从模型返回的评分

        st.success(f"ESG评分: {score}")


if __name__ == "__main__":
    main()
