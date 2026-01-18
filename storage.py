import os
import json
import time
from typing import List, Tuple, Optional

import aiomysql
import redis.asyncio as redis


class Storage:
    def __init__(self):
        self.redis = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        self.mysql_pool: Optional[aiomysql.Pool] = None
        self.ctx_ttl = int(os.getenv("CTX_TTL_SECONDS", "604800"))
        self.max_turns = int(os.getenv("CTX_MAX_TURNS", "10"))

    async def init_mysql(self):
        self.mysql_pool = await aiomysql.create_pool(
            host=os.environ["MYSQL_HOST"],
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.environ["MYSQL_USER"],
            password=os.environ["MYSQL_PASSWORD"],
            db=os.environ["MYSQL_DB"],
            autocommit=True,
            minsize=1,
            maxsize=10,
        )

    # -------- Redis: context --------
    def _ctx_key(self, user_id: int) -> str:
        return f"ctx:{user_id}"

    async def ctx_get(self, user_id: int) -> List[Tuple[str, str]]:
        items = await self.redis.lrange(self._ctx_key(user_id), 0, -1)
        out: List[Tuple[str, str]] = []
        for s in items:
            obj = json.loads(s)
            out.append((obj["role"], obj["text"]))
        return out

    async def ctx_append(self, user_id: int, role: str, text: str) -> None:
        key = self._ctx_key(user_id)
        await self.redis.rpush(key, json.dumps({"role": role, "text": text}, ensure_ascii=False))
        await self.redis.expire(key, self.ctx_ttl)

        max_len = self.max_turns * 2
        cur_len = await self.redis.llen(key)
        if cur_len > max_len:
            await self.redis.ltrim(key, cur_len - max_len, -1)

    async def ctx_clear(self, user_id: int) -> None:
        await self.redis.delete(self._ctx_key(user_id))

    # -------- Redis: rate limit (zset sliding window) --------
    def _rl_key(self, user_id: int) -> str:
        return f"rl:{user_id}"

    async def rate_limit_ok(self, user_id: int, limit: int, window_sec: int) -> bool:
        key = self._rl_key(user_id)
        now = time.time()
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window_sec)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_sec)
        _, _, count, _ = await pipe.execute()
        return int(count) <= limit

    # -------- Redis: lock --------
    def _lock_key(self, user_id: int) -> str:
        return f"lock:{user_id}"

    async def acquire_lock(self, user_id: int, ttl_sec: int = 30) -> bool:
        return bool(await self.redis.set(self._lock_key(user_id), "1", nx=True, ex=ttl_sec))

    async def release_lock(self, user_id: int) -> None:
        await self.redis.delete(self._lock_key(user_id))

    # -------- MySQL --------
    async def ensure_user(self, user_id: int, username: str | None, first_name: str | None, model: str) -> None:
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO users(user_id, username, first_name, model)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      username=VALUES(username),
                      first_name=VALUES(first_name),
                      model=VALUES(model)
                    """,
                    (user_id, username, first_name, model),
                )

    async def save_message(self, user_id: int, role: str, text: str) -> None:
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO messages(user_id, role, text) VALUES (%s, %s, %s)",
                    (user_id, role, text),
                )

    async def get_summary(self, user_id: int) -> Optional[str]:
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT summary FROM summaries WHERE user_id=%s", (user_id,))
                row = await cur.fetchone()
                return row[0] if row else None

    async def set_summary(self, user_id: int, summary: str) -> None:
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO summaries(user_id, summary) VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE summary=VALUES(summary)
                    """,
                    (user_id, summary),
                )

    async def add_fact(self, user_id: int, fact: str, confidence: int = 70) -> None:
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO user_facts(user_id, fact, confidence) VALUES (%s, %s, %s)",
                    (user_id, fact, confidence),
                )

    async def list_facts(self, user_id: int, limit: int = 10) -> List[str]:
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT fact FROM user_facts WHERE user_id=%s ORDER BY id DESC LIMIT %s",
                    (user_id, limit),
                )
                rows = await cur.fetchall()
                return [r[0] for r in rows]
