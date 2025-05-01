import asyncio
from handlers.lead.bot_instance import bot, dp
import logics.starting

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot, skip_updates=True))
