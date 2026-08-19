#!/usr/bin/env python3
"""Edit-regime suite: high-copy-fraction prompts (the agent workload).
Reuses bench_dflash's measurement code, swaps the prompt set."""
import bench_dflash as B

SRC_GO = '''package cache

import (
	"errors"
	"sync"
	"time"
)

var ErrNotFound = errors.New("cache: key not found")

type entry struct {
	value     []byte
	expiresAt time.Time
}

type Cache struct {
	mu      sync.RWMutex
	items   map[string]entry
	ttl     time.Duration
	maxSize int
}

func New(ttl time.Duration, maxSize int) *Cache {
	return &Cache{
		items:   make(map[string]entry),
		ttl:     ttl,
		maxSize: maxSize,
	}
}

func (c *Cache) Get(key string) ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	e, ok := c.items[key]
	if !ok {
		return nil, ErrNotFound
	}
	if time.Now().After(e.expiresAt) {
		return nil, ErrNotFound
	}
	return e.value, nil
}

func (c *Cache) Set(key string, value []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.items) >= c.maxSize {
		c.evictOldest()
	}
	c.items[key] = entry{
		value:     value,
		expiresAt: time.Now().Add(c.ttl),
	}
}

func (c *Cache) evictOldest() {
	var oldestKey string
	var oldest time.Time
	first := true
	for k, e := range c.items {
		if first || e.expiresAt.Before(oldest) {
			oldestKey = k
			oldest = e.expiresAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(c.items, oldestKey)
	}
}

func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}
'''

SRC_PY = '''import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class Record:
    key: str
    payload: dict[str, Any]
    version: int = 1


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        self._data: dict[str, Record] = {}
        self._dirty = False

    def load(self) -> None:
        try:
            with open(self.path) as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            logger.warning("store %s missing, starting empty", self.path)
            return
        for key, item in raw.items():
            self._data[key] = Record(key=key, payload=item["payload"], version=item["version"])

    def save(self) -> None:
        if not self._dirty:
            return
        out = {k: {"payload": r.payload, "version": r.version} for k, r in self._data.items()}
        with open(self.path, "w") as fh:
            json.dump(out, fh, indent=2)
        self._dirty = False

    def put(self, key: str, payload: dict[str, Any]) -> Record:
        existing = self._data.get(key)
        version = existing.version + 1 if existing else 1
        rec = Record(key=key, payload=payload, version=version)
        self._data[key] = rec
        self._dirty = True
        return rec

    def get(self, key: str) -> Record | None:
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._dirty = True
            return True
        return False

    def keys(self) -> Iterable[str]:
        return self._data.keys()
'''

B.PROMPTS = {
 "edit_go_field": "Here is a Go file:\n\n```go\n" + SRC_GO + "```\n\nAdd a `hits` and `misses` counter to the Cache struct, increment them in Get, and add a `Stats() (hits, misses int)` method. Output the COMPLETE modified file, unchanged parts included, nothing else.",
 "edit_go_rename": "Here is a Go file:\n\n```go\n" + SRC_GO + "```\n\nRename the `entry` type to `cacheEntry` everywhere, and rename `maxSize` to `capacity`. Change nothing else. Output the COMPLETE modified file, nothing else.",
 "edit_py_logging": "Here is a Python file:\n\n```python\n" + SRC_PY + "```\n\nAdd a debug log line at the start of `put`, `get` and `delete` recording the key. Change nothing else. Output the COMPLETE modified file, nothing else.",
 "edit_py_type": "Here is a Python file:\n\n```python\n" + SRC_PY + "```\n\nAdd an `updated_at: float` field to Record (default 0.0), set it with time.time() in `put`, and import time. Output the COMPLETE modified file, nothing else.",
}

if __name__ == "__main__":
    B.main()
