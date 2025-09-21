---
layout: post
title: "React Hooks 指南"
date: 2024-01-15
categories: [编程, React]
tags: [React, Hooks, JavaScript, 前端开发]
image: /assets/img/react-hooks.jpg
description: "React Hooks 是 React 16.8 引入的新特性，它让我们可以在函数组件中使用状态和其他 React 特性。"
---

React Hooks 是 React 16.8 引入的新特性，它让我们可以在函数组件中使用状态和其他 React 特性。

## 📋 目录

- [useState](#usestate)
- [useEffect](#useeffect)
- [useContext](#usecontext)
- [useReducer](#usereducer)
- [自定义 Hooks](#自定义-hooks)
- [最佳实践](#最佳实践)

## useState

`useState` 是最基础的 Hook，用于在函数组件中添加状态。

### 基本用法

```javascript
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>当前计数: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        增加
      </button>
    </div>
  );
}
```

### 函数式更新

```javascript
const [count, setCount] = useState(0);

// 使用函数式更新
setCount(prevCount => prevCount + 1);
```

## useEffect

`useEffect` 用于处理副作用，相当于类组件中的 `componentDidMount`、`componentDidUpdate` 和 `componentWillUnmount`。

### 基本用法

```javascript
import React, { useState, useEffect } from 'react';

function DataFetcher() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 组件挂载时执行
    fetchData();
    
    // 清理函数
    return () => {
      // 组件卸载时执行清理
    };
  }, []); // 空依赖数组，只在挂载时执行

  const fetchData = async () => {
    try {
      const response = await fetch('/api/data');
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('获取数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>加载中...</div>;
  
  return <div>{JSON.stringify(data)}</div>;
}
```

### 依赖数组

```javascript
// 每次渲染都执行
useEffect(() => {
  console.log('每次渲染都执行');
});

// 只在挂载时执行
useEffect(() => {
  console.log('只在挂载时执行');
}, []);

// 当 count 变化时执行
useEffect(() => {
  console.log('count 变化了:', count);
}, [count]);
```

## useContext

`useContext` 用于在组件树中共享数据，避免 prop drilling。

### 创建 Context

```javascript
import React, { createContext, useContext } from 'react';

const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function ThemedButton() {
  const { theme, setTheme } = useContext(ThemeContext);
  
  return (
    <button 
      style={{ 
        backgroundColor: theme === 'light' ? '#fff' : '#333',
        color: theme === 'light' ? '#333' : '#fff'
      }}
      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
    >
      切换主题
    </button>
  );
}
```

## useReducer

`useReducer` 是 `useState` 的替代方案，适用于复杂的状态逻辑。

### 基本用法

```javascript
import React, { useReducer } from 'react';

const initialState = { count: 0 };

function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { count: state.count + 1 };
    case 'decrement':
      return { count: state.count - 1 };
    case 'reset':
      return initialState;
    default:
      throw new Error();
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <div>
      <p>计数: {state.count}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>
        +
      </button>
      <button onClick={() => dispatch({ type: 'decrement' })}>
        -
      </button>
      <button onClick={() => dispatch({ type: 'reset' })}>
        重置
      </button>
    </div>
  );
}
```

## 自定义 Hooks

自定义 Hooks 让我们可以提取组件逻辑到可重用的函数中。

### 示例：useLocalStorage

```javascript
import { useState, useEffect } from 'react';

function useLocalStorage(key, initialValue) {
  // 获取初始值
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });

  // 设置值并保存到 localStorage
  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
}

// 使用自定义 Hook
function MyComponent() {
  const [name, setName] = useLocalStorage('name', '');

  return (
    <input
      type="text"
      value={name}
      onChange={(e) => setName(e.target.value)}
      placeholder="输入你的名字"
    />
  );
}
```

## 最佳实践

### 1. 只在顶层调用 Hooks

```javascript
// ❌ 错误：在条件语句中调用
function MyComponent({ condition }) {
  if (condition) {
    const [state, setState] = useState(0); // 错误！
  }
}

// ✅ 正确：在顶层调用
function MyComponent({ condition }) {
  const [state, setState] = useState(0);
  
  if (condition) {
    // 在条件语句中使用状态
  }
}
```

### 2. 合理使用依赖数组

```javascript
// ❌ 错误：缺少依赖
useEffect(() => {
  fetchData(userId);
}, []); // 缺少 userId 依赖

// ✅ 正确：包含所有依赖
useEffect(() => {
  fetchData(userId);
}, [userId]);
```

### 3. 使用 useCallback 和 useMemo 优化性能

```javascript
import React, { useState, useCallback, useMemo } from 'react';

function ExpensiveComponent({ items, filter }) {
  // 使用 useMemo 缓存计算结果
  const filteredItems = useMemo(() => {
    return items.filter(item => item.name.includes(filter));
  }, [items, filter]);

  // 使用 useCallback 缓存函数
  const handleClick = useCallback((id) => {
    console.log('点击了项目:', id);
  }, []);

  return (
    <div>
      {filteredItems.map(item => (
        <div key={item.id} onClick={() => handleClick(item.id)}>
          {item.name}
        </div>
      ))}
    </div>
  );
}
```

## 总结

React Hooks 让函数组件更加强大和灵活，通过合理使用各种 Hooks，我们可以：

- 简化组件逻辑
- 提高代码复用性
- 更好的性能优化
- 更清晰的代码结构

记住 Hooks 的使用规则，合理组织代码，就能充分发挥 React Hooks 的优势。

---

*最后更新：2024-01-15*
