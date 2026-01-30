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

    def _cnt_key(self, user_id: int) -> str:
        return f"cnt:{user_id}"
    
    def _require_int_user_id(self, user_id):
        if not isinstance(user_id, int):
            raise TypeError(f"user_id must be int, got {type(user_id)}: {user_id!r}")


    async def inc_user_msg_count(self, user_id: int, ttc_sec: int = 2592000) -> int:
        """
        Счётчик сообщений пользователя. TTL = 30 дней по умолчанию.
        Возвращает текущее значение после инкремента.
        """
        self._require_int_user_id(user_id)


        key = self._cnt_key(user_id)
        val = await self.redis.incr(key)
        await self.redis.expire(key, ttc_sec)
        return int(val)

    async def ctx_get(self, user_id: int) -> List[Tuple[str, str]]:
        self._require_int_user_id(user_id)

        items = await self.redis.lrange(self._ctx_key(user_id), 0, -1)
        out: List[Tuple[str, str]] = []
        for s in items:
            obj = json.loads(s)
            out.append((obj["role"], obj["text"]))
        return out

    async def ctx_append(self, user_id: int, role: str, text: str) -> None:
        self._require_int_user_id(user_id)
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
        self._require_int_user_id(user_id)
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
        self._require_int_user_id(user_id)
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
        self._require_int_user_id(user_id)
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO messages(user_id, role, text) VALUES (%s, %s, %s)",
                    (user_id, role, text),
                )

    async def get_summary(self, user_id: int) -> Optional[str]:
        self._require_int_user_id(user_id)
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT summary FROM summaries WHERE user_id=%s", (user_id,))
                row = await cur.fetchone()
                return row[0] if row else None

    async def set_summary(self, user_id: int, summary: str) -> None:
        self._require_int_user_id(user_id)
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

    async def last_messages(self, user_id: int, limit: int = 40) -> list[tuple[str, str]]:
        self._require_int_user_id(user_id)
        assert self.mysql_pool

        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT role, text
                    FROM messages
                    WHERE user_id=%s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                    
                )
                rows = await cur.fetchall()
        rows.reverse()
        return [(r[0], r[1]) for r in rows]

    async def add_fact(self, user_id: int, fact: str, confidence: int = 70) -> None:
        self._require_int_user_id(user_id)

        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO user_facts(user_id, fact, confidence) VALUES (%s, %s, %s)",
                    (user_id, fact, confidence),
                )

    async def list_facts(self, user_id: int, limit: int = 10) -> List[str]:
        self._require_int_user_id(user_id)

        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT fact FROM user_facts WHERE user_id=%s ORDER BY id DESC LIMIT %s",
                    (user_id, limit),
                )
                rows = await cur.fetchall()
                return [r[0] for r in rows]

    async def clear_long_memory(self, user_id: int) -> None:
        self._require_int_user_id(user_id)

        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM summaries WHERE user_id=%s", (user_id,))
                await cur.execute("DELETE FROM user_facts WHERE user_id=%s", (user_id,))

    async def touch_user(self, user_id: int):
        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE users SET last_seen=NOW() WHERE user_id=%s",
                    (user_id,),
                )

    async def inc_messages(self, user_id: int):
        self._require_int_user_id(user_id)

        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE users SET messages_count=messages_count+1 WHERE user_id=%s",
                    (user_id,),
                )

    async def log_event(self, user_id: int, event_type: str, meta: dict | None = None):
        self._require_int_user_id(user_id)

        assert self.mysql_pool
        async with self.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO events(user_id, event_type, meta) VALUES (%s, %s, %s)",
                    (user_id, event_type, json.dump(meta) if meta else None),
                )

