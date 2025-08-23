#!/usr/bin/env python

import streamlit as st
import asyncio
import platform
from typing import Any
import os

# Configure environment before MetaGPT imports
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "SERPAPI_API_KEY" in st.secrets:
    os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

# Set required MetaGPT environment variables
os.environ.setdefault("METAGPT_TEXT_TO_IMAGE_MODEL_URL", "")
os.environ.setdefault("METAGPT_TEXT_TO_SPEECH_MODEL_URL", "")
os.environ.setdefault("WORKSPACE_ROOT", "/tmp")

try:
    from metagpt.actions import Action, UserRequirement
    from metagpt.logs import logger
    from metagpt.roles import Role
    from metagpt.schema import Message
    from research_actions import CollectLinks, WebBrowseAndSummarize, ConductResearch
    from metagpt.roles.role import RoleReactMode
    METAGPT_AVAILABLE = True
except ImportError as e:
    METAGPT_AVAILABLE = False
    st.error(f"MetaGPT not available: {str(e)}")


class Researcher(Role):
    """Researcher role with three actions"""
    name: str = "Researcher"
    profile: str = "Research Assistant"
    
    def __init__(self, **data):
        super().__init__(**data)
        self.set_actions([CollectLinks, WebBrowseAndSummarize, ConductResearch])
        self._set_react_mode(RoleReactMode.BY_ORDER.value, len(self.actions))
    
    async def research_topic(self, topic: str) -> str:
        """Conduct quick research on a topic"""
        # Collect links (limited to 2 queries, 2 URLs each)
        collect_action = CollectLinks()
        links = await collect_action.run(topic, decomposition_nums=2, url_per_query=2)
        
        # Browse and summarize (max 4 URLs total)
        browse_action = WebBrowseAndSummarize()
        summaries = []
        url_count = 0
        for query, urls in links.items():
            if urls and url_count < 4:
                remaining_urls = 4 - url_count
                limited_urls = urls[:remaining_urls]
                result = await browse_action.run(*limited_urls, query=query)
                summaries.extend(result.values())
                url_count += len(limited_urls)
        
        # Conduct research
        research_action = ConductResearch()
        content = "\n---\n".join(summaries)
        report = await research_action.run(topic, content)
        return report


class RequestResearch(Action):
    """Action: Request research from researcher"""
    
    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    You are {name}, a {profile} preparing for a debate on: {topic}
    
    ## TASK
    What specific aspect of this topic would you like to research to strengthen your position?
    Provide ONE specific research query (1-2 sentences) that would help you in this debate.
    Focus on facts, statistics, examples, or evidence that would support your {profile} perspective.
    """
    name: str = "RequestResearch"

    async def run(self, name: str, profile: str, topic: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(name=name, profile=profile, topic=topic)
        rsp = await self._aask(prompt)
        return rsp


class SpeakAloud(Action):
    """Action: Speak out aloud in a debate (quarrel)"""

    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name}, you are in a debate with {opponent_name1} and {opponent_name2}. You are debating the topic:
    {idea}
    ## DEBATE HISTORY
    Previous rounds:
    {context}
    ## RESEARCH INFORMATION
    {research_info}
    ## YOUR TURN
    {instruction}
    """
    name: str = "SpeakAloud"

    async def run(self, context: str, name: str, opponent_name1: str, opponent_name2: str, idea: str = "", profile: str = "", round_num: int = 1, research_info: str = "") -> str:
        if round_num <= 3:
            instruction = f"This is round {round_num} of 3 opening rounds. You should ONLY state your view on the topic, give your arguments and how you logically and rigorously arrived at your views. Do NOT rebut or respond to any of your opponents' arguments. Your viewpoint should be clear, concise, and stereotypical of a {profile}. MANDATORY: Use specific facts, statistics, and evidence from your research information to support your arguments. Include proper citations in your response using [Source: URL or description] format."
        else:
            instruction = f"This is round {round_num}. You should first restate your view, then closely respond to your opponents' latest arguments, defend your arguments, and attack your opponents' arguments if they differ from yours. Craft a strong, logically rigorous response in {name}'s rhetoric and viewpoints. MANDATORY: Support your arguments with specific evidence from your research and include citations using [Source: URL or description] format."
        
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, opponent_name1=opponent_name1, opponent_name2=opponent_name2, idea=idea, profile=profile, instruction=instruction, research_info=research_info)
        rsp = await self._aask(prompt)
        return rsp


class EvaluateDebate(Action):
    """Action: Evaluate and summarize a debate into concise recommendations"""

    PROMPT_TEMPLATE: str = """
    ## ROLE
    You are a neutral evaluator analyzing a debate to provide clear, actionable recommendations.

    ## DEBATE TOPIC
    {topic}

    ## DEBATE CONTENT
    {debate_content}

    ## TASK
    Write a concise evaluation (200-300 words) that:
    1. Summarizes key arguments from all participants
    2. Identifies core trade-offs and considerations
    3. Provides balanced, practical recommendations for decision-makers
    4. Focuses on actionable solutions

    Your response should be structured and help stakeholders make informed decisions.
    """
    
    name: str = "EvaluateDebate"

    async def run(self, topic: str, debate_content: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(topic=topic, debate_content=debate_content)
        rsp = await self._aask(prompt)
        return rsp


class ProvideAdvice(Action):
    """Action: Provide compromise solutions with consequences analysis"""

    PROMPT_TEMPLATE: str = """
    ## ROLE
    You are a neutral advisor analyzing a debate to provide compromise solutions.

    ## DEBATE TOPIC
    {topic}

    ## EVALUATION
    {evaluation}

    ## TASK
    Based on the evaluation, provide 3 compromise solutions that balance all perspectives.
    For each solution:
    1. **Solution**: Brief description
    2. **Benefits**: How it addresses each party's concerns
    3. **Consequences**: Potential negative outcomes for each party
    4. **Implementation**: Practical steps

    Focus on realistic compromises that all parties could accept.
    """
    
    name: str = "ProvideAdvice"

    async def run(self, topic: str, evaluation: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(topic=topic, evaluation=evaluation)
        rsp = await self._aask(prompt)
        return rsp


class DebateEvaluator(Role):
    name: str = "Evaluator"
    profile: str = "Neutral Analyst"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([EvaluateDebate])

    async def evaluate(self, topic: str, debate_messages: list) -> str:
        todo = EvaluateDebate()
        debate_content = "\n\n".join(f"{msg.sent_from}: {msg.content}" for msg in debate_messages)
        evaluation = await todo.run(topic=topic, debate_content=debate_content)
        return evaluation


class DebateAdvisor(Role):
    name: str = "Advisor"
    profile: str = "Compromise Specialist"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([ProvideAdvice])

    async def advise(self, topic: str, evaluation: str) -> str:
        todo = ProvideAdvice()
        advice = await todo.run(topic=topic, evaluation=evaluation)
        return advice


class Debator(Role):
    name: str = ""
    profile: str = ""
    opponent_name1: str = ""
    opponent_name2: str = ""
    research_info: str = ""
    research_count: int = 0
    max_research: int = 1

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SpeakAloud])
        self._watch([UserRequirement, SpeakAloud])
    
    async def request_research(self, topic: str, researcher: Researcher) -> str:
        """Request research from the researcher"""
        if self.research_count >= self.max_research:
            return "Research limit reached"
            
        request_action = RequestResearch()
        query = await request_action.run(name=self.name, profile=self.profile, topic=topic)
        
        # Get research report
        research_result = await researcher.research_topic(query)
        
        self.research_info += f"\n\nResearch Query: {query}\nResearch Result: {research_result}"
        self.research_count += 1
        return research_result

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if self.name in msg.send_to or msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self) -> Message:
        todo = self.rc.todo
        memories = self.get_memories()
        context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        
        topic = self.get_memories()[0].content if self.get_memories() else "debate topic"
        speaker_turns = len([m for m in memories if m.sent_from == self.name])
        
        rsp = await todo.run(context=context, name=self.name, opponent_name1=self.opponent_name1, opponent_name2=self.opponent_name2, idea=topic, profile=self.profile, round_num=speaker_turns + 1, research_info=self.research_info)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to={self.opponent_name1, self.opponent_name2}
        )
        self.rc.memory.add(msg)
        return msg


async def run_debate(idea: str, n_round: int = 5):
    """Run a debate and return results"""
    School = Debator(name="Principal", profile="School", opponent_name1="John", opponent_name2="Mom")
    Student = Debator(name="John", profile="Student", opponent_name1="Mom", opponent_name2="Principal")
    Parent = Debator(name="Mom", profile="Parent", opponent_name1="Principal", opponent_name2="John")
    
    # Create researcher
    researcher = Researcher()
    
    # Research phase - each debater gets 1 research only
    debaters = [School, Student, Parent]
    research_log = []
    
    for debater in debaters:
        research_result = await debater.request_research(idea, researcher)
        research_log.append({
            "debater": debater.name,
            "research": research_result
        })
    
    # Start with Principal responding to the topic
    current_speaker = School
    second_speaker = Student
    third_speaker = Parent
    
    # Store all debate messages for evaluation
    all_messages = []
    debate_log = []
    
    # Initial message to start the debate
    msg = Message(content=idea, role="user", send_to={"Principal"}, sent_from="User")
    
    for round_num in range(n_round):
        # Current speaker responds
        response = await current_speaker.run(msg)
        
        debate_log.append({
            "round": round_num + 1,
            "speaker": current_speaker.name,
            "content": response.content
        })
        
        # Store the response for evaluation
        all_messages.append(response)
        
        # Switch speakers for next round
        current_speaker, second_speaker, third_speaker = third_speaker, current_speaker, second_speaker
        msg = response
    
    # Evaluate the debate
    evaluator = DebateEvaluator()
    evaluation = await evaluator.evaluate(idea, all_messages)
    
    # Generate advice
    advisor = DebateAdvisor()
    advice = await advisor.advise(idea, evaluation)
    
    return debate_log, evaluation, research_log, advice


# Check API keys and MetaGPT availability
def check_requirements():
    if not METAGPT_AVAILABLE:
        return False, "MetaGPT framework is not available."
    
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OpenAI API key is required but not configured."
    
    return True, "All requirements met."

# Streamlit UI
st.set_page_config(
    page_title="AI Debate Platform",
    page_icon="🗣️",
    layout="wide"
)

st.title("🗣️ AI Debate Platform")
st.markdown("Enter a debate topic and watch AI agents discuss different perspectives!")

# Check requirements
requirements_ok, requirements_msg = check_requirements()
if not requirements_ok:
    st.error(f"⚠️ {requirements_msg}")
    st.info("This app requires proper API key configuration. Please contact the administrator.")
    st.stop()

# Input section
with st.form("debate_form"):
    topic = st.text_input(
        "Debate Topic:",
        placeholder="e.g., Should schools take phones after bedtime?",
        help="Enter any topic you'd like the AI agents to debate"
    )
    
    n_rounds = st.slider("Number of rounds:", min_value=3, max_value=10, value=6)
    
    submitted = st.form_submit_button("Start Debate", type="primary")

if submitted and topic:
    with st.spinner("🤖 AI agents are debating..."):
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        try:
            # Add progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Starting debate...")
            progress_bar.progress(10)
            
            debate_log, evaluation, research_log, advice = asyncio.run(run_debate(topic, n_rounds))
            
            progress_bar.progress(100)
            status_text.text("Debate completed!")
            
            # Show advisor recommendations first
            st.subheader("💡 Advisor Recommendations")
            st.info(advice)
            
            # Show evaluation
            st.subheader("📋 Debate Evaluation")
            st.success(evaluation)
            
            # Show research phase
            with st.expander("🔍 Research Phase", expanded=False):
                for entry in research_log:
                    st.markdown(f"**{entry['debater']} Research:**")
                    st.write(entry['research'])
                    st.divider()
            
            # Show debate process in expandable section
            with st.expander("🗣️ View Full Debate Process", expanded=False):
                for entry in debate_log:
                    with st.container():
                        st.markdown(f"**Round {entry['round']} - {entry['speaker']}:**")
                        st.write(entry['content'])
                        st.divider()
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please try again with a different topic or contact support if the issue persists.")

elif submitted and not topic:
    st.warning("Please enter a debate topic!")

# Instructions
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Enter a topic** - Any controversial or discussion-worthy topic for **school policy** only
    2. **Choose rounds** - More rounds = deeper discussion
    3. **Start debate** - Three AI agents will first research, then debate from different perspectives:
       - 🏫 **Principal** (School perspective)
       - 👨‍🎓 **John** (Student perspective) 
       - 👩‍👧 **Mom** (Parent perspective)
    4. **View results** - Get a summary evaluation and see the full debate transcript
    """)
    
    st.header("Example Topics")
    st.markdown("""
    - Should schools ban smartphones?
    - Is remote work better than office work?
    - Should social media have age restrictions?
    - Is homework necessary for learning?
    """)