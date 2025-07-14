import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- إعدادات الصفحة ---
st.set_page_config(page_title="فريق التسويق الذكي لوكالتك", page_icon="🚀", layout="wide")

# --- العنوان الرئيسي وشرح الأداة ---
st.title("🚀 فريق التسويق الذكي المخصص لوكالتك")
st.markdown("""
أدخل فكرة المنتج، ثم **زود الفريق بمعلومات عن العميل والأسلوب المطلوب**، وسيقوم بإنشاء تقرير تسويقي متكامل ومخصص.
""")

# --- إعداد البيئة والمفاتيح (تم التعديل ليتوافق مع Streamlit Cloud) ---
@st.cache_resource
def setup_environment_and_models():
    try:
        # **التعديل الحاسم**: نقرأ المفاتيح مباشرة من st.secrets الخاص بـ Streamlit
        groq_api_key = st.secrets["GROQ_API_KEY"]
        tavily_api_key = st.secrets["TAVILY_API_KEY"]

        # نقوم بتعيين المفاتيح كمتغيرات بيئة لتتوافق مع مكتبة Tavily
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        search_tool = TavilySearchResults(max_results=5)
        # نمرر مفتاح Groq مباشرة عند إنشاء النموذج
        llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0.4)

        return search_tool, llm
    except Exception as e:
        # رسالة خطأ أوضح إذا لم يتم العثور على المفاتيح
        st.error(f"🛑 خطأ في إعداد البيئة: {e}. هل قمت بإضافة GROQ_API_KEY و TAVILY_API_KEY في إعدادات التطبيق على Streamlit Cloud؟")
        return None, None

search_tool, llm = setup_environment_and_models()

if not search_tool or not llm:
    st.stop()

# --- بناء سلاسل العمل (Chains) مع الأخذ في الاعتبار السياق الجديد ---
@st.cache_resource
def get_marketing_chains(_search_tool, _llm):
    # المرحلة 1: سلسلة أبحاث السوق (تستخدم معلومات العميل لتركيز البحث)
    research_prompt = ChatPromptTemplate.from_template(
        """أنت خبير أبحاث سوق يعمل لصالح وكالة تسويق.
        **معلومات العميل:** {client_info}
        **المهمة:** بناءً على معلومات العميل، قم بإنشاء ملخص تنفيذي شامل عن "{question}" يغطي: الجمهور المستهدف، المنافسين الرئيسيين، والترندات الحالية.
        **نتائج البحث الأولية:** {context}

        **الملخص التنفيذي المخصص للعميل:**"""
    )
    research_chain = (
        RunnablePassthrough.assign(context=lambda x: _search_tool.invoke(x["question"]))
        | research_prompt | _llm | StrOutputParser()
    )

    # المرحلة 2: سلسلة تطوير الاستراتيجية (تستخدم معلومات العميل)
    strategy_prompt = ChatPromptTemplate.from_template(
        """أنت خبير استراتيجيات تسويق عبقري.
        **معلومات العميل:** {client_info}
        **المهمة:** لقد استلمت ملخصًا تنفيذيًا. قم بتحليله لإنشاء وثيقة استراتيجية مخصصة لهذا العميل تحتوي على: تحليل SWOT، شخصية العميل المثالي (Customer Persona)، وثلاث رسائل تسويقية رئيسية.
        **الملخص التنفيذي:** {research_summary}

        **الوثيقة الاستراتيجية المخصصة:**"""
    )
    strategy_chain = strategy_prompt | _llm | StrOutputParser()

    # المرحلة 3: سلسلة إنشاء المحتوى (تستخدم معلومات العميل وأسلوب المحتوى)
    content_creator_prompt = ChatPromptTemplate.from_template(
        """أنت كاتب محتوى إعلاني (Copywriter) محترف في وكالة تسويق.
        **معلومات العميل:** {client_info}
        **أسلوب المحتوى المطلوب (Tone of Voice):** {tone_of_voice}
        **المهمة:** لقد استلمت وثيقة استراتيجية. مهمتك هي تحويلها إلى محتوى تسويقي إبداعي **يتوافق تمامًا مع أسلوب المحتوى المطلوب**.
        أنشئ التالي: 3 منشورات سوشيال ميديا، سيناريو فيديو قصير، ومسودة بريد إلكتروني.
        **الوثيقة الاستراتيجية:** {strategy_document}

        **المحتوى التسويقي المخصص للعميل:**"""
    )
    content_creator_chain = content_creator_prompt | _llm | StrOutputParser()

    return research_chain, strategy_chain, content_creator_chain

research_chain, strategy_chain, content_creator_chain = get_marketing_chains(search_tool, llm)

# --- واجهة المستخدم المحدثة ---
st.sidebar.header("📝 أدخل تفاصيل المهمة")

topic = st.sidebar.text_input("1. ما هو المنتج أو فكرة الحملة؟", placeholder="مثال: خصومات الصيف على العطور")
client_info = st.sidebar.text_area("2. من هو العميل؟ (صفه باختصار)", placeholder="مثال: 'عطور الفخامة'، متجر عطور راقٍ يستهدف الرجال والنساء فوق 30 عامًا.")
tone_of_voice = st.sidebar.text_area("3. ما هو أسلوب المحتوى المطلوب؟", placeholder="مثال: أسلوب فخم وراقٍ، يستخدم كلمات مثل 'حصرية'، 'تألق'، 'جاذبية'. تجنب الأسلوب الشبابي والمرح.")

if st.sidebar.button("🚀 أطلق فريق التسويق المخصص!"):
    if topic and client_info and tone_of_voice:
        st.sidebar.info("✅ تم استلام الطلب. الفريق المخصص يعمل الآن...")

        # --- تشغيل خط التجميع الكامل مع تمرير السياق الجديد ---
        with st.spinner("المرحلة 1: خبير الأبحاث يركز بحثه على العميل..."):
            research_summary = research_chain.invoke({
                "question": f"أبحاث السوق لمنتج: {topic}",
                "client_info": client_info
            })

        with st.spinner("المرحلة 2: خبير الاستراتيجية يبني خطة مخصصة..."):
            strategy_document = strategy_chain.invoke({
                "research_summary": research_summary,
                "client_info": client_info
            })

        with st.spinner("المرحلة 3: خبير المحتوى يكتب بأسلوب العميل..."):
            marketing_content = content_creator_chain.invoke({
                "strategy_document": strategy_document,
                "client_info": client_info,
                "tone_of_voice": tone_of_voice
            })

        st.sidebar.success("🎉 اكتملت المهمة المخصصة بنجاح!")

        # --- عرض النتائج النهائية ---
        st.header(f"📊 تقرير التسويق المخصص لـ: {topic}")

        tab1, tab2, tab3 = st.tabs(["الأبحاث 📈", "الاستراتيجية 🎯", "المحتوى ✍️"])

        with tab1:
            st.subheader("الملخص التنفيذي للأبحاث")
            st.markdown(research_summary)

        with tab2:
            st.subheader("الوثيقة الاستراتيجية")
            st.markdown(strategy_document)

        with tab3:
            st.subheader("المحتوى التسويقي الجاهز (بأسلوب العميل)")
            st.markdown(marketing_content)

    else:
        st.sidebar.warning("يرجى ملء جميع الخانات الثلاث لبدء العمل.")
