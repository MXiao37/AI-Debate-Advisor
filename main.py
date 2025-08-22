"""
Filename: MetaGPT/examples/debate.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.1.3 of RFC 116, modify the data type of the `send_to`
        value of the `Message` object; modify the argument type of `get_by_actions`.
"""

import asyncio
import platform
from typing import Any

import fire

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from research_actions import CollectLinks, WebBrowseAndSummarize, ConductResearch
from metagpt.roles import Role
from metagpt.roles.role import RoleReactMode
import asyncio
import re






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
            instruction = f"This is round {round_num} of 3 opening rounds. You should ONLY state your view on the topic, give your arguments and how you logically and rigorously arrived at your views. Do NOT rebut or respond to any of your opponents' arguments yet. Your viewpoint should be clear, concise, and extremely stereotypical of a {profile}. MANDATORY: Use specific facts, statistics, and evidence from your research information to support your arguments. Include proper citations in your response using [Source: URL or description] format."
        else:
            instruction = f"This is round {round_num}. You should defend your arguments, and attack and directly rebut your opponents' arguments if they differ from yours. Craft a strong, logically rigorous response in {name}'s rhetoric and viewpoints. MANDATORY: Support your arguments with specific evidence from your research and include citations using [Source: URL or description] format."
        
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
        # accept messages sent (from opponent) to self, disregard own messages from the last round
        self.rc.news = [msg for msg in self.rc.news if self.name in msg.send_to or msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        memories = self.get_memories()
        context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        
        # Get the debate topic from the first message
        topic = self.get_memories()[0].content if self.get_memories() else "debate topic"
        # Count how many times this speaker has spoken
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


async def debate(idea: str, investment: float = 3.0, n_round: int = 5):
    """Run a debate and evaluate the results"""
    School = Debator(name="Principal", profile="School", opponent_name1="John", opponent_name2 = "Mom")
    Student = Debator(name="John", profile="Student", opponent_name1="Mom", opponent_name2 = "Principal")
    Parent = Debator(name="Mom", profile="Parent", opponent_name1="Principal", opponent_name2 = "John")
    
    # Create researcher
    researcher = Researcher()
    
    logger.info(f"Starting debate on: {idea}")
    
    # Research phase - each debater gets 1 research only
    debaters = [School, Student, Parent]
    logger.info(f"\n=== Research Phase ===")
    for debater in debaters:
        logger.info(f"{debater.name} requesting research...")
        await debater.request_research(idea, researcher)
    
    # Start with Principal responding to the topic
    current_speaker = School
    second_speaker = Student
    third_speaker = Parent
    
    # Store all debate messages for evaluation
    all_messages = []
    
    # Initial message to start the debate
    msg = Message(content=idea, role="user", send_to={"Principal"}, sent_from="User")
    
    for round_num in range(n_round):
        logger.info(f"\n=== Round {round_num + 1} ===\n{current_speaker.name}'s turn:")
        
        # Current speaker responds
        response = await current_speaker.run(msg)
        logger.info(f"{current_speaker.name}: {response.content}")
        
        # Store the response for evaluation
        all_messages.append(response)
        
        # Switch speakers for next round
        current_speaker, second_speaker, third_speaker = third_speaker, current_speaker, second_speaker
        msg = response
    
    # Evaluate the debate
    logger.info("\n=== EVALUATING DEBATE ===")
    evaluator = DebateEvaluator()
    evaluation = await evaluator.evaluate(idea, all_messages)
    logger.info(f"\n=== EVALUATION RESULTS ===\n{evaluation}")
    
    return evaluation


def main(idea: str, investment: float = 3.0, n_round: int = 6):
    """
    :param idea: Debate topic, such as "Topic: The U.S. should commit more in climate change fighting"
                 or "Trump: Climate change is a hoax"
    :param investment: contribute a certain dollar amount to watch the debate
    :param n_round: maximum rounds of the debate
    :return:
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debate(idea, investment, n_round))


if __name__ == "__main__":
    fire.Fire(main)  # run as python debate.py --idea="TOPIC" --investment=3.0 --n_round=5