import asyncio
from metagpt.roles.researcher import RESEARCH_PATH, Researcher


async def main():
    topic = "dataiku vs. datarobot"
    role = Researcher(language="en-us")
    
    # Add delay to avoid rate limiting
    await asyncio.sleep(2)
    
    await role.run(topic)
    print(f"save report to {RESEARCH_PATH / f'{topic}.md'}.")


if __name__ == "__main__":
    asyncio.run(main())