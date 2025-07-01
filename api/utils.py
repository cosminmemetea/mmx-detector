from fastapi_limiter.depends import RateLimiter


class RateLimitDependency:
    def __init__(self, rate_limit: str):
        count_str, period = rate_limit.split("/")
        self.times = int(count_str)
        if period == "second":
            self.seconds = 1
        elif period == "minute":
            self.seconds = 60
        elif period == "hour":
            self.seconds = 3600
        else:
            raise ValueError(f"Unsupported rate period: {period}")

    async def __call__(self):
        limiter = RateLimiter(times=self.times, seconds=self.seconds)
        await limiter()
