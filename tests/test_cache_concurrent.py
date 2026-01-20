"""
趋势雷达选股系统 - 缓存和并发模块测试
测试优化后的缓存管理器、限流器、并发工具等
"""
import pytest
import sys
import os
import time
import pandas as pd
import numpy as np
import tempfile
import threading
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cache_manager_optimized import (
    CacheManager,
    LRUCache,
    cached
)
from core.utils_optimized import (
    RateLimiter,
    ConcurrentRateLimiter,
    ProgressTracker,
    ThreadPool,
    BatchProcessor,
    retry_on_failure
)


class TestLRUCache:
    """测试LRU缓存"""

    def test_lru_basic(self):
        """测试LRU基本功能"""
        cache = LRUCache(capacity=3)

        # 添加数据
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')

        assert cache.size() == 3

        # 获取数据
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'

    def test_lru_eviction(self):
        """测试LRU淘汰"""
        cache = LRUCache(capacity=2)

        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')  # 应该淘汰key1

        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'
        assert cache.size() == 2

    def test_lru_clear(self):
        """测试清空LRU缓存"""
        cache = LRUCache(capacity=10)

        cache.put('key1', 'value1')
        cache.put('key2', 'value2')

        cache.clear()

        assert cache.size() == 0
        assert cache.get('key1') is None


class TestCacheManager:
    """测试优化的缓存管理器"""

    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """创建缓存管理器实例"""
        return CacheManager(
            cache_dir=temp_cache_dir,
            memory_cache_size=5,
            enable_compression=True
        )

    def test_memory_cache_hit(self, cache_manager):
        """测试内存缓存命中"""
        test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        cache_manager.put('test', test_data, 'key1')
        result = cache_manager.get('test', 'key1', ttl_days=7)

        assert result is not None
        assert len(result) == 3
        # 应该命中内存缓存
        stats = cache_manager.get_stats()
        assert stats['memory_hits'] > 0

    def test_disk_cache(self, cache_manager):
        """测试磁盘缓存"""
        test_data = {'key': 'value'}

        cache_manager.put('test', test_data, 'key2')
        # 清空内存缓存
        cache_manager.memory_cache.clear()

        result = cache_manager.get('test', 'key2', ttl_days=7)

        assert result is not None
        assert result == test_data
        # 应该命中磁盘缓存
        stats = cache_manager.get_stats()
        assert stats['disk_hits'] > 0

    def test_cache_expiration(self, cache_manager):
        """测试缓存过期"""
        test_data = 'test data'

        cache_manager.put('test', test_data, 'key3')
        cache_manager.memory_cache.clear()

        # 设置过期时间为-1天(已过期)
        result = cache_manager.get('test', 'key3', ttl_days=-1)

        assert result is None

    def test_cache_stats(self, cache_manager):
        """测试缓存统计"""
        test_data = pd.DataFrame({'a': [1, 2, 3]})

        # 写入缓存
        cache_manager.put('test', test_data, 'key4')
        cache_manager.put('test', test_data, 'key5')

        # 读取缓存
        cache_manager.get('test', 'key4', ttl_days=7)
        cache_manager.get('test', 'nonexistent', ttl_days=7)

        stats = cache_manager.get_stats()

        assert stats['writes'] == 2
        assert stats['memory_hits'] >= 1
        assert stats['misses'] >= 1

    def test_clear_cache(self, cache_manager):
        """测试清理缓存"""
        test_data = pd.DataFrame({'a': [1, 2, 3]})

        cache_manager.put('test', test_data, 'key6')
        cache_manager.clear('test')

        result = cache_manager.get('test', 'key6', ttl_days=7)
        assert result is None


class TestRateLimiter:
    """测试限流器"""

    def test_rate_limiter_basic(self):
        """测试基本限流功能"""
        limiter = RateLimiter(max_calls_per_minute=10)

        start_time = time.time()

        # 快速调用10次
        for _ in range(10):
            limiter.wait_if_needed()

        elapsed = time.time() - start_time

        # 应该不会等待
        assert elapsed < 1.0

        # 第11次调用应该触发等待
        limiter.wait_if_needed()
        elapsed_after = time.time() - start_time

        # 应该等待接近6秒(因为前10次调用在0秒内完成)
        assert elapsed_after >= 5.0

    def test_rate_limiter_thread_safe(self):
        """测试线程安全"""
        limiter = RateLimiter(max_calls_per_minute=5)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(2):
                    limiter.wait_if_needed()
                    results.append(time.time())
            except Exception as e:
                errors.append(e)

        # 创建多个线程
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 应该没有错误
        assert len(errors) == 0
        # 应该有6个结果
        assert len(results) == 6


class TestConcurrentRateLimiter:
    """测试并发限流器"""

    def test_token_bucket_basic(self):
        """测试令牌桶基本功能"""
        limiter = ConcurrentRateLimiter(max_rate=60.0, capacity=60)

        # 应该能连续获取60个令牌
        for _ in range(60):
            assert limiter.acquire(timeout=1.0) == True

        # 第61个令牌应该需要等待
        start = time.time()
        assert limiter.acquire(timeout=2.0) == True
        elapsed = time.time() - start

        # 应该等待大约1秒
        assert elapsed >= 0.8

    def test_token_bucket_refill(self):
        """测试令牌补充"""
        limiter = ConcurrentRateLimiter(max_rate=60.0, capacity=10)

        # 获取所有令牌
        for _ in range(10):
            limiter.acquire(timeout=1.0)

        # 等待1秒，应该补充约1个令牌
        time.sleep(1.1)

        # 应该能再获取1个令牌
        assert limiter.acquire(timeout=1.0) == True


class TestProgressTracker:
    """测试进度追踪器"""

    def test_progress_tracker_basic(self, capsys):
        """测试基本进度追踪"""
        progress = ProgressTracker(total=100, desc="Test")

        for i in range(10):
            progress.update(10)

        progress.finish()

        captured = capsys.readouterr()
        assert "完成" in captured.out

    def test_progress_tracker_no_verbose(self, capsys):
        """测试不显示进度"""
        progress = ProgressTracker(total=100, desc="Test", verbose=False)

        progress.update(10)
        progress.finish()

        captured = capsys.readouterr()
        # 不应该有进度输出
        assert "]" not in captured.out


class TestThreadPool:
    """测试线程池"""

    def test_thread_pool_submit(self):
        """测试提交任务"""
        pool = ThreadPool(max_workers=2)

        def task(x):
            return x * 2

        future = pool.submit(task, 5)
        result = future.result(timeout=5.0)

        assert result == 10

        pool.shutdown()

    def test_thread_pool_map(self):
        """测试批量处理"""
        pool = ThreadPool(max_workers=3)

        def task(x):
            time.sleep(0.1)
            return x * 2

        results = pool.map(task, [1, 2, 3, 4, 5])

        assert results == [2, 4, 6, 8, 10]

        pool.shutdown()


class TestBatchProcessor:
    """测试批量处理器"""

    def test_batch_processor_basic(self):
        """测试批量处理"""
        processor = BatchProcessor(batch_size=3, max_workers=2, progress_bar=False)

        def process_item(item):
            return item * 2

        items = list(range(10))
        results = processor.process(items, process_item)

        # 结果可能乱序，只检查长度
        assert len(results) == 10
        # 检查每个结果是否正确
        for result in results:
            assert result % 2 == 0

    def test_batch_processor_batches(self):
        """测试按批处理"""
        processor = BatchProcessor(batch_size=4, max_workers=2, progress_bar=False)

        def process_batch(batch):
            return [x * 2 for x in batch]

        items = list(range(10))
        results = processor.process_batches(items, process_batch)

        # 检查结果
        assert len(results) == 10
        # 检查每个结果是否正确
        for result in results:
            assert result % 2 == 0


class TestRetryDecorator:
    """测试重试装饰器"""

    def test_retry_on_success(self):
        """测试成功时不需要重试"""
        call_count = [0]

        @retry_on_failure(max_retries=3, delay=0.1)
        def func():
            call_count[0] += 1
            return "success"

        result = func()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        """测试失败时重试"""
        call_count = [0]

        @retry_on_failure(max_retries=3, delay=0.05)
        def func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"

        result = func()

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhausted(self):
        """测试重试次数用尽"""
        @retry_on_failure(max_retries=2, delay=0.05)
        def func():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            func()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
