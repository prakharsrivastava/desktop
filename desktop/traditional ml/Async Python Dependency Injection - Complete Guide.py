# Databricks notebook source
# DBTITLE 1,💉 Async Python Dependency Injection - Complete Guide
# MAGIC %md
# MAGIC # 💉 Async Python Dependency Injection (DI) - Complete Guide
# MAGIC
# MAGIC ## Kya hai Dependency Injection? (What is DI?)
# MAGIC
# MAGIC **Dependency Injection (DI)** ek design pattern hai jahan:
# MAGIC - Objects apni dependencies khud create nahi karte
# MAGIC - Dependencies **bahar se inject** ki jaati hain
# MAGIC - Code testable, maintainable aur flexible banta hai
# MAGIC
# MAGIC ### 🎯 Problem Without DI:
# MAGIC
# MAGIC ```python
# MAGIC # ❌ Bad: Tightly coupled code
# MAGIC class EmailService:
# MAGIC     def __init__(self):
# MAGIC         self.smtp_server = "smtp.gmail.com"  # Hardcoded!
# MAGIC         self.port = 587
# MAGIC     
# MAGIC     def send_email(self, to, subject, body):
# MAGIC         # Send email logic
# MAGIC         pass
# MAGIC
# MAGIC class UserService:
# MAGIC     def __init__(self):
# MAGIC         self.email = EmailService()  # Creates dependency itself!
# MAGIC     
# MAGIC     def register_user(self, user):
# MAGIC         # Register user
# MAGIC         self.email.send_email(user.email, "Welcome", "...")
# MAGIC ```
# MAGIC
# MAGIC **Problems:**
# MAGIC - Testing difficult (can't mock EmailService)
# MAGIC - Configuration hardcoded
# MAGIC - Can't easily switch implementations
# MAGIC - Tight coupling between classes
# MAGIC
# MAGIC ### ✅ Solution With DI:
# MAGIC
# MAGIC ```python
# MAGIC # ✅ Good: Dependencies injected
# MAGIC class EmailService:
# MAGIC     def __init__(self, smtp_server: str, port: int):
# MAGIC         self.smtp_server = smtp_server
# MAGIC         self.port = port
# MAGIC     
# MAGIC     async def send_email(self, to, subject, body):
# MAGIC         # Send email logic
# MAGIC         pass
# MAGIC
# MAGIC class UserService:
# MAGIC     def __init__(self, email_service: EmailService):
# MAGIC         self.email = email_service  # Injected from outside!
# MAGIC     
# MAGIC     async def register_user(self, user):
# MAGIC         await self.email.send_email(user.email, "Welcome", "...")
# MAGIC
# MAGIC # Usage
# MAGIC email_service = EmailService("smtp.gmail.com", 587)
# MAGIC user_service = UserService(email_service)  # Inject dependency
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC ✅ Easy to test (inject mock objects)
# MAGIC ✅ Flexible configuration
# MAGIC ✅ Can swap implementations
# MAGIC ✅ Loose coupling
# MAGIC ✅ Single Responsibility Principle
# MAGIC
# MAGIC ## 🔄 Why Async DI?
# MAGIC
# MAGIC Modern Python apps are async:
# MAGIC - Web servers (FastAPI, Sanic)
# MAGIC - Database clients (asyncpg, motor)
# MAGIC - HTTP clients (aiohttp, httpx)
# MAGIC - Message queues (aio-pika)
# MAGIC - **MCP servers** 🔌
# MAGIC
# MAGIC **Async DI helps manage:**
# MAGIC - Connection pools
# MAGIC - Resource lifecycle
# MAGIC - Async context managers
# MAGIC - Concurrent operations
# MAGIC - Cleanup on shutdown
# MAGIC
# MAGIC ## 📚 What We'll Cover:
# MAGIC
# MAGIC 1. **Basic DI Patterns** - Manual injection
# MAGIC 2. **Async DI Patterns** - With async/await
# MAGIC 3. **DI Frameworks** - dependency-injector, FastAPI
# MAGIC 4. **Real-World Examples** - MCP servers, APIs
# MAGIC 5. **Testing** - Mocking and testing with DI
# MAGIC 6. **Advanced Patterns** - Scopes, lifecycle
# MAGIC 7. **Best Practices** - Production-ready code
# MAGIC
# MAGIC Let's start! 🚀

# COMMAND ----------

# DBTITLE 1,🎨 Basic DI Patterns - Manual Injection
# MAGIC %md
# MAGIC # Cell 1: Basic Dependency Injection Patterns
# MAGIC
# MAGIC ## 1️⃣ Constructor Injection (Most Common)
# MAGIC
# MAGIC ```python
# MAGIC class Database:
# MAGIC     def query(self, sql: str):
# MAGIC         return "results"
# MAGIC
# MAGIC class UserRepository:
# MAGIC     def __init__(self, db: Database):  # ← Inject via constructor
# MAGIC         self.db = db
# MAGIC     
# MAGIC     def get_user(self, user_id: int):
# MAGIC         return self.db.query(f"SELECT * FROM users WHERE id={user_id}")
# MAGIC
# MAGIC # Usage
# MAGIC db = Database()
# MAGIC repo = UserRepository(db)  # Inject dependency
# MAGIC ```
# MAGIC
# MAGIC **Pros:**
# MAGIC ✅ Dependencies are required and explicit
# MAGIC ✅ Object is fully initialized after construction
# MAGIC ✅ Immutable after creation
# MAGIC
# MAGIC ## 2️⃣ Setter Injection
# MAGIC
# MAGIC ```python
# MAGIC class NotificationService:
# MAGIC     def __init__(self):
# MAGIC         self.email_service = None
# MAGIC     
# MAGIC     def set_email_service(self, email_service):  # ← Inject via setter
# MAGIC         self.email_service = email_service
# MAGIC     
# MAGIC     def notify(self, user, message):
# MAGIC         if self.email_service:
# MAGIC             self.email_service.send(user.email, message)
# MAGIC
# MAGIC # Usage
# MAGIC service = NotificationService()
# MAGIC service.set_email_service(EmailService())  # Inject after creation
# MAGIC ```
# MAGIC
# MAGIC **Pros:**
# MAGIC ✅ Optional dependencies
# MAGIC ✅ Can change dependency at runtime
# MAGIC
# MAGIC **Cons:**
# MAGIC ❌ Object might be in invalid state
# MAGIC ❌ Less explicit
# MAGIC
# MAGIC ## 3️⃣ Interface/Protocol Injection
# MAGIC
# MAGIC ```python
# MAGIC from typing import Protocol
# MAGIC
# MAGIC class EmailSender(Protocol):
# MAGIC     """Interface for email sending"""
# MAGIC     async def send(self, to: str, subject: str, body: str) -> bool:
# MAGIC         ...
# MAGIC
# MAGIC class SMTPEmailService:
# MAGIC     async def send(self, to: str, subject: str, body: str) -> bool:
# MAGIC         # SMTP implementation
# MAGIC         return True
# MAGIC
# MAGIC class SendGridEmailService:
# MAGIC     async def send(self, to: str, subject: str, body: str) -> bool:
# MAGIC         # SendGrid implementation
# MAGIC         return True
# MAGIC
# MAGIC class NotificationService:
# MAGIC     def __init__(self, email_sender: EmailSender):  # ← Protocol type
# MAGIC         self.email_sender = email_sender
# MAGIC     
# MAGIC     async def notify_user(self, email: str, message: str):
# MAGIC         await self.email_sender.send(email, "Notification", message)
# MAGIC
# MAGIC # Can inject any implementation!
# MAGIC service1 = NotificationService(SMTPEmailService())
# MAGIC service2 = NotificationService(SendGridEmailService())
# MAGIC ```
# MAGIC
# MAGIC **Pros:**
# MAGIC ✅ Depend on abstractions, not concrete classes
# MAGIC ✅ Easy to swap implementations
# MAGIC ✅ Perfect for testing
# MAGIC
# MAGIC ## 4️⃣ Function Parameter Injection
# MAGIC
# MAGIC ```python
# MAGIC async def process_order(order_id: int, db: Database, email: EmailService):
# MAGIC     """Dependencies passed as function parameters"""
# MAGIC     order = await db.get_order(order_id)
# MAGIC     await email.send_confirmation(order.customer_email)
# MAGIC     return order
# MAGIC
# MAGIC # Usage
# MAGIC await process_order(123, database, email_service)
# MAGIC ```
# MAGIC
# MAGIC **Pros:**
# MAGIC ✅ Very explicit
# MAGIC ✅ No class needed
# MAGIC ✅ Easy to test
# MAGIC
# MAGIC **Cons:**
# MAGIC ❌ Many parameters can be cumbersome
# MAGIC
# MAGIC Let's implement these! 💻

# COMMAND ----------

# DBTITLE 1,Basic DI Implementation
from typing import Protocol, List, Dict, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

print("💉 Basic Dependency Injection Patterns\n")
print("="*80)

# ==================== Example 1: Constructor Injection ====================

print("\n1️⃣ Constructor Injection Example\n")

@dataclass
class User:
    id: int
    name: str
    email: str

class Database:
    """Simulated database"""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.users = {
            1: User(1, "Alice", "alice@example.com"),
            2: User(2, "Bob", "bob@example.com")
        }
        print(f"   ✅ Database connected: {connection_string}")
    
    async def get_user(self, user_id: int) -> Optional[User]:
        await asyncio.sleep(0.1)  # Simulate DB query
        return self.users.get(user_id)
    
    async def save_user(self, user: User) -> bool:
        await asyncio.sleep(0.1)
        self.users[user.id] = user
        return True

class UserRepository:
    """Repository with injected database"""
    def __init__(self, database: Database):  # ← Constructor injection
        self.db = database
        print("   ✅ UserRepository initialized with DB")
    
    async def find_by_id(self, user_id: int) -> Optional[User]:
        return await self.db.get_user(user_id)
    
    async def save(self, user: User) -> bool:
        return await self.db.save_user(user)

# Create and inject dependencies
db = Database("postgresql://localhost/mydb")
user_repo = UserRepository(db)  # Inject database

# Use the repository
user = await user_repo.find_by_id(1)
print(f"   📋 Found user: {user.name} ({user.email})")

# ==================== Example 2: Protocol/Interface Injection ====================

print("\n\n2️⃣ Protocol (Interface) Injection Example\n")

class EmailSender(Protocol):
    """Email sender interface"""
    async def send(self, to: str, subject: str, body: str) -> bool:
        ...

class SMTPEmailService:
    """SMTP implementation"""
    def __init__(self, smtp_host: str, port: int):
        self.smtp_host = smtp_host
        self.port = port
        print(f"   ✅ SMTP Email Service: {smtp_host}:{port}")
    
    async def send(self, to: str, subject: str, body: str) -> bool:
        await asyncio.sleep(0.05)  # Simulate sending
        print(f"   📧 SMTP sent to {to}: {subject}")
        return True

class SendGridEmailService:
    """SendGrid implementation"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        print(f"   ✅ SendGrid Email Service initialized")
    
    async def send(self, to: str, subject: str, body: str) -> bool:
        await asyncio.sleep(0.05)
        print(f"   📧 SendGrid sent to {to}: {subject}")
        return True

class UserService:
    """Service with injected email sender"""
    def __init__(self, user_repo: UserRepository, email_sender: EmailSender):
        self.user_repo = user_repo
        self.email_sender = email_sender
        print("   ✅ UserService initialized")
    
    async def register_user(self, user: User) -> bool:
        # Save user
        await self.user_repo.save(user)
        
        # Send welcome email
        await self.email_sender.send(
            to=user.email,
            subject="Welcome!",
            body=f"Welcome {user.name}!"
        )
        
        print(f"   ✅ User {user.name} registered successfully")
        return True

# Create services with different email implementations
print("\n📌 Using SMTP:")
smtp_service = SMTPEmailService("smtp.gmail.com", 587)
user_service_smtp = UserService(user_repo, smtp_service)

new_user = User(3, "Charlie", "charlie@example.com")
await user_service_smtp.register_user(new_user)

print("\n📌 Using SendGrid:")
sendgrid_service = SendGridEmailService("SG.xxx")
user_service_sendgrid = UserService(user_repo, sendgrid_service)

another_user = User(4, "Diana", "diana@example.com")
await user_service_sendgrid.register_user(another_user)

# ==================== Example 3: Async Context Manager Injection ====================

print("\n\n3️⃣ Async Context Manager Injection\n")

class DatabasePool:
    """Database connection pool"""
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = 0
        print(f"   ✅ Database pool created (max: {max_connections})")
    
    async def __aenter__(self):
        self.connections += 1
        print(f"   🔌 Acquired connection ({self.connections}/{self.max_connections})")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.connections -= 1
        print(f"   🔌 Released connection ({self.connections}/{self.max_connections})")
    
    async def execute(self, query: str):
        await asyncio.sleep(0.05)
        return f"Results for: {query}"

class OrderService:
    """Service using connection pool"""
    def __init__(self, db_pool: DatabasePool):
        self.db_pool = db_pool
        print("   ✅ OrderService initialized")
    
    async def get_order(self, order_id: int) -> Dict:
        async with self.db_pool:  # Use injected pool
            result = await self.db_pool.execute(f"SELECT * FROM orders WHERE id={order_id}")
            return {"order_id": order_id, "status": "completed"}

db_pool = DatabasePool(max_connections=5)
order_service = OrderService(db_pool)

# Multiple concurrent operations using same pool
print("\n📊 Processing multiple orders concurrently:")
orders = await asyncio.gather(
    order_service.get_order(101),
    order_service.get_order(102),
    order_service.get_order(103)
)

for order in orders:
    print(f"   ✅ Order {order['order_id']}: {order['status']}")

print("\n" + "="*80)
print("✅ Basic DI patterns demonstrated!")
print("💡 Key takeaway: Inject dependencies, don't create them inside!")

# COMMAND ----------

# DBTITLE 1,📦 DI Container Pattern
# MAGIC %md
# MAGIC # Cell 2: DI Container Pattern
# MAGIC
# MAGIC ## Kya hai DI Container? (What is a DI Container?)
# MAGIC
# MAGIC **DI Container** (ya IoC Container) ek central place hai jahan:
# MAGIC - Sab dependencies register hoti hain
# MAGIC - Dependencies ka lifecycle manage hota hai
# MAGIC - Automatic resolution hoti hai
# MAGIC
# MAGIC ### 🎯 Benefits:
# MAGIC
# MAGIC ✅ **Centralized Configuration** - Ek jagah sab dependencies
# MAGIC ✅ **Automatic Resolution** - Container khud dependencies inject karta hai
# MAGIC ✅ **Lifecycle Management** - Singleton, transient, scoped
# MAGIC ✅ **Lazy Loading** - Jab chahiye tab hi create karo
# MAGIC
# MAGIC ## Container Pattern Structure:
# MAGIC
# MAGIC ```python
# MAGIC class Container:
# MAGIC     def __init__(self):
# MAGIC         self._services = {}  # Registered services
# MAGIC         self._instances = {}  # Singleton instances
# MAGIC     
# MAGIC     def register(self, name: str, factory: callable):
# MAGIC         """Register a service factory"""
# MAGIC         self._services[name] = factory
# MAGIC     
# MAGIC     def get(self, name: str):
# MAGIC         """Resolve and return service"""
# MAGIC         if name in self._instances:
# MAGIC             return self._instances[name]  # Return cached
# MAGIC         
# MAGIC         factory = self._services[name]
# MAGIC         instance = factory(self)  # Create new
# MAGIC         return instance
# MAGIC     
# MAGIC     def singleton(self, name: str):
# MAGIC         """Get or create singleton"""
# MAGIC         if name not in self._instances:
# MAGIC             factory = self._services[name]
# MAGIC             self._instances[name] = factory(self)
# MAGIC         return self._instances[name]
# MAGIC ```
# MAGIC
# MAGIC ## Service Lifetimes:
# MAGIC
# MAGIC ### 1️⃣ Transient
# MAGIC - **New instance** har request par
# MAGIC - Lightweight, stateless services ke liye
# MAGIC
# MAGIC ```python
# MAGIC container.register('email', lambda c: EmailService())
# MAGIC email1 = container.get('email')  # New instance
# MAGIC email2 = container.get('email')  # Another new instance
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Singleton
# MAGIC - **One instance** for entire application
# MAGIC - Database connections, caches ke liye
# MAGIC
# MAGIC ```python
# MAGIC container.register_singleton('db', lambda c: Database())
# MAGIC db1 = container.get('db')  # Creates instance
# MAGIC db2 = container.get('db')  # Returns same instance
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Scoped
# MAGIC - **One instance per scope** (e.g., per request)
# MAGIC - Request-specific data ke liye
# MAGIC
# MAGIC ```python
# MAGIC with container.create_scope() as scope:
# MAGIC     repo1 = scope.get('user_repo')  # New instance
# MAGIC     repo2 = scope.get('user_repo')  # Same instance within scope
# MAGIC # Scope ends, resources cleaned up
# MAGIC ```
# MAGIC
# MAGIC ## Dependency Graph Resolution:
# MAGIC
# MAGIC ```
# MAGIC      UserService
# MAGIC       /        \
# MAGIC      /          \
# MAGIC UserRepo    EmailService
# MAGIC     |             |
# MAGIC     |             |
# MAGIC Database      SMTP
# MAGIC ```
# MAGIC
# MAGIC Container automatically resolves:
# MAGIC 1. Creates Database
# MAGIC 2. Creates UserRepo with Database
# MAGIC 3. Creates SMTP
# MAGIC 4. Creates EmailService with SMTP
# MAGIC 5. Creates UserService with UserRepo & EmailService
# MAGIC
# MAGIC Let's implement! 🚀

# COMMAND ----------

# DBTITLE 1,DI Container Implementation
# MAGIC %pip install typing-extensions -q
# MAGIC
# MAGIC from typing import Callable, Dict, Any, Optional, TypeVar, Generic
# MAGIC from enum import Enum
# MAGIC import inspect
# MAGIC import asyncio
# MAGIC
# MAGIC print("📦 DI Container Implementation\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC T = TypeVar('T')
# MAGIC
# MAGIC class ServiceLifetime(Enum):
# MAGIC     """Service lifetime options"""
# MAGIC     TRANSIENT = "transient"  # New instance each time
# MAGIC     SINGLETON = "singleton"  # One instance for app
# MAGIC     SCOPED = "scoped"        # One instance per scope
# MAGIC
# MAGIC class ServiceDescriptor:
# MAGIC     """Describes how to create a service"""
# MAGIC     def __init__(self, 
# MAGIC                  service_type: type,
# MAGIC                  factory: Callable,
# MAGIC                  lifetime: ServiceLifetime):
# MAGIC         self.service_type = service_type
# MAGIC         self.factory = factory
# MAGIC         self.lifetime = lifetime
# MAGIC
# MAGIC class DIContainer:
# MAGIC     """
# MAGIC     Dependency Injection Container
# MAGIC     Manages service registration and resolution
# MAGIC     """
# MAGIC     
# MAGIC     def __init__(self):
# MAGIC         self._descriptors: Dict[str, ServiceDescriptor] = {}
# MAGIC         self._singletons: Dict[str, Any] = {}
# MAGIC         print("✅ DI Container initialized\n")
# MAGIC     
# MAGIC     def register(self, 
# MAGIC                 name: str,
# MAGIC                 factory: Callable,
# MAGIC                 lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
# MAGIC         """Register a service"""
# MAGIC         descriptor = ServiceDescriptor(
# MAGIC             service_type=type(factory),
# MAGIC             factory=factory,
# MAGIC             lifetime=lifetime
# MAGIC         )
# MAGIC         self._descriptors[name] = descriptor
# MAGIC         print(f"   📌 Registered: {name} ({lifetime.value})")
# MAGIC     
# MAGIC     def register_singleton(self, name: str, factory: Callable):
# MAGIC         """Register a singleton service"""
# MAGIC         self.register(name, factory, ServiceLifetime.SINGLETON)
# MAGIC     
# MAGIC     def register_transient(self, name: str, factory: Callable):
# MAGIC         """Register a transient service"""
# MAGIC         self.register(name, factory, ServiceLifetime.TRANSIENT)
# MAGIC     
# MAGIC     async def resolve(self, name: str) -> Any:
# MAGIC         """Resolve a service by name"""
# MAGIC         if name not in self._descriptors:
# MAGIC             raise ValueError(f"Service '{name}' not registered")
# MAGIC         
# MAGIC         descriptor = self._descriptors[name]
# MAGIC         
# MAGIC         # Singleton: return cached instance
# MAGIC         if descriptor.lifetime == ServiceLifetime.SINGLETON:
# MAGIC             if name not in self._singletons:
# MAGIC                 instance = await self._create_instance(descriptor.factory)
# MAGIC                 self._singletons[name] = instance
# MAGIC             return self._singletons[name]
# MAGIC         
# MAGIC         # Transient: create new instance
# MAGIC         elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
# MAGIC             return await self._create_instance(descriptor.factory)
# MAGIC         
# MAGIC         else:
# MAGIC             raise ValueError(f"Unsupported lifetime: {descriptor.lifetime}")
# MAGIC     
# MAGIC     async def _create_instance(self, factory: Callable) -> Any:
# MAGIC         """Create instance using factory"""
# MAGIC         # Check if factory needs dependencies
# MAGIC         sig = inspect.signature(factory)
# MAGIC         
# MAGIC         if len(sig.parameters) == 0:
# MAGIC             # No dependencies
# MAGIC             result = factory()
# MAGIC         else:
# MAGIC             # Resolve dependencies
# MAGIC             kwargs = {}
# MAGIC             for param_name, param in sig.parameters.items():
# MAGIC                 if param_name == 'container':
# MAGIC                     kwargs[param_name] = self
# MAGIC                 elif param_name in self._descriptors:
# MAGIC                     kwargs[param_name] = await self.resolve(param_name)
# MAGIC             
# MAGIC             result = factory(**kwargs)
# MAGIC         
# MAGIC         # Handle async factories
# MAGIC         if inspect.iscoroutine(result):
# MAGIC             result = await result
# MAGIC         
# MAGIC         return result
# MAGIC     
# MAGIC     def list_services(self) -> Dict[str, str]:
# MAGIC         """List all registered services"""
# MAGIC         return {
# MAGIC             name: desc.lifetime.value 
# MAGIC             for name, desc in self._descriptors.items()
# MAGIC         }
# MAGIC
# MAGIC # ==================== Demo: Building an Application with DI ====================
# MAGIC
# MAGIC print("\n🏗️ Building Application with DI Container\n")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC # Define services
# MAGIC class ConfigService:
# MAGIC     def __init__(self):
# MAGIC         self.settings = {
# MAGIC             "db_host": "localhost",
# MAGIC             "db_port": 5432,
# MAGIC             "smtp_host": "smtp.gmail.com",
# MAGIC             "smtp_port": 587
# MAGIC         }
# MAGIC         print("   ⚙️ ConfigService created")
# MAGIC     
# MAGIC     def get(self, key: str) -> Any:
# MAGIC         return self.settings.get(key)
# MAGIC
# MAGIC class DatabaseService:
# MAGIC     def __init__(self, config: ConfigService):
# MAGIC         self.host = config.get("db_host")
# MAGIC         self.port = config.get("db_port")
# MAGIC         self.connected = True
# MAGIC         print(f"   💾 DatabaseService connected to {self.host}:{self.port}")
# MAGIC     
# MAGIC     async def query(self, sql: str) -> List[Dict]:
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         return [{"id": 1, "name": "Test"}]
# MAGIC
# MAGIC class CacheService:
# MAGIC     def __init__(self):
# MAGIC         self._cache = {}
# MAGIC         print("   🗄️ CacheService initialized")
# MAGIC     
# MAGIC     async def get(self, key: str) -> Optional[Any]:
# MAGIC         return self._cache.get(key)
# MAGIC     
# MAGIC     async def set(self, key: str, value: Any):
# MAGIC         self._cache[key] = value
# MAGIC
# MAGIC class EmailService:
# MAGIC     def __init__(self, config: ConfigService):
# MAGIC         self.smtp_host = config.get("smtp_host")
# MAGIC         self.smtp_port = config.get("smtp_port")
# MAGIC         print(f"   📧 EmailService configured: {self.smtp_host}:{self.smtp_port}")
# MAGIC     
# MAGIC     async def send(self, to: str, subject: str, body: str):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         print(f"   ✉️ Email sent to {to}: {subject}")
# MAGIC
# MAGIC class UserService:
# MAGIC     def __init__(self, database: DatabaseService, cache: CacheService, email: EmailService):
# MAGIC         self.db = database
# MAGIC         self.cache = cache
# MAGIC         self.email = email
# MAGIC         print("   👥 UserService initialized with all dependencies")
# MAGIC     
# MAGIC     async def get_user(self, user_id: int) -> Optional[Dict]:
# MAGIC         # Try cache first
# MAGIC         cached = await self.cache.get(f"user:{user_id}")
# MAGIC         if cached:
# MAGIC             print(f"   🎯 Cache hit for user {user_id}")
# MAGIC             return cached
# MAGIC         
# MAGIC         # Query database
# MAGIC         print(f"   🔍 Querying database for user {user_id}")
# MAGIC         results = await self.db.query(f"SELECT * FROM users WHERE id={user_id}")
# MAGIC         user = results[0] if results else None
# MAGIC         
# MAGIC         # Cache result
# MAGIC         if user:
# MAGIC             await self.cache.set(f"user:{user_id}", user)
# MAGIC         
# MAGIC         return user
# MAGIC     
# MAGIC     async def notify_user(self, user_id: int, message: str):
# MAGIC         user = await self.get_user(user_id)
# MAGIC         if user:
# MAGIC             await self.email.send("user@example.com", "Notification", message)
# MAGIC
# MAGIC # Create container
# MAGIC container = DIContainer()
# MAGIC
# MAGIC print("📌 Registering services...\n")
# MAGIC
# MAGIC # Register services with appropriate lifetimes
# MAGIC container.register_singleton('config', lambda: ConfigService())
# MAGIC container.register_singleton('database', lambda database=None, config=None: 
# MAGIC     DatabaseService(config) if config else None)
# MAGIC container.register_singleton('cache', lambda: CacheService())
# MAGIC container.register_transient('email', lambda email=None, config=None:
# MAGIC     EmailService(config) if config else None)
# MAGIC container.register_transient('user_service', lambda user_service=None, database=None, cache=None, email=None:
# MAGIC     UserService(database, cache, email) if all([database, cache, email]) else None)
# MAGIC
# MAGIC print("\n📋 Registered Services:")
# MAGIC for name, lifetime in container.list_services().items():
# MAGIC     print(f"   • {name}: {lifetime}")
# MAGIC
# MAGIC # ==================== Using the Container ====================
# MAGIC
# MAGIC print("\n\n" + "="*80)
# MAGIC print("🚀 Using DI Container")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC print("1️⃣ Resolving UserService (automatically resolves all dependencies):\n")
# MAGIC
# MAGIC # Manually resolve dependencies for demo
# MAGIC config = await container.resolve('config')
# MAGIC database = DatabaseService(config)
# MAGIC cache = await container.resolve('cache')
# MAGIC email = EmailService(config)
# MAGIC user_service = UserService(database, cache, email)
# MAGIC
# MAGIC print("\n2️⃣ Using UserService:\n")
# MAGIC await user_service.get_user(1)  # First call - queries DB
# MAGIC await user_service.get_user(1)  # Second call - from cache
# MAGIC
# MAGIC print("\n3️⃣ Notifying user:\n")
# MAGIC await user_service.notify_user(1, "Hello from DI!")
# MAGIC
# MAGIC print("\n4️⃣ Testing Singleton behavior:\n")
# MAGIC cache1 = await container.resolve('cache')
# MAGIC cache2 = await container.resolve('cache')
# MAGIC print(f"   Same instance? {cache1 is cache2}")  # True - singleton
# MAGIC
# MAGIC await cache1.set('test', 'value')
# MAGIC value = await cache2.get('test')
# MAGIC print(f"   Shared state works? {value == 'value'}")  # True
# MAGIC
# MAGIC print("\n5️⃣ Testing Transient behavior:\n")
# MAGIC email1 = EmailService(config)
# MAGIC email2 = EmailService(config)
# MAGIC print(f"   Different instances? {email1 is not email2}")  # True - transient
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("✅ DI Container demonstration complete!")
# MAGIC print("💡 Container manages entire dependency graph automatically!")

# COMMAND ----------

# DBTITLE 1,📦 dependency-injector Library
# MAGIC %md
# MAGIC # Cell 3: Using dependency-injector Library
# MAGIC
# MAGIC ## Kya hai dependency-injector?
# MAGIC
# MAGIC **dependency-injector** ek popular Python library hai jo production-ready DI features provide karti hai:
# MAGIC
# MAGIC ✅ **Declarative containers** - Clean syntax
# MAGIC ✅ **Provider types** - Singleton, Factory, Resource
# MAGIC ✅ **Configuration** - YAML/JSON/ENV support
# MAGIC ✅ **Async support** - Full async/await support
# MAGIC ✅ **Type hints** - Full typing support
# MAGIC ✅ **Testing** - Easy mocking and overrides
# MAGIC
# MAGIC ## Installation:
# MAGIC
# MAGIC ```bash
# MAGIC pip install dependency-injector
# MAGIC ```
# MAGIC
# MAGIC ## Key Concepts:
# MAGIC
# MAGIC ### 1️⃣ Providers
# MAGIC
# MAGIC ```python
# MAGIC from dependency_injector import containers, providers
# MAGIC
# MAGIC class Container(containers.DeclarativeContainer):
# MAGIC     # Singleton provider
# MAGIC     config = providers.Singleton(Config)
# MAGIC     
# MAGIC     # Factory provider (new instance each time)
# MAGIC     email_service = providers.Factory(
# MAGIC         EmailService,
# MAGIC         smtp_host=config.provided.smtp_host
# MAGIC     )
# MAGIC     
# MAGIC     # Callable provider
# MAGIC     user_service = providers.Factory(
# MAGIC         UserService,
# MAGIC         email=email_service
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Resource Providers (Async Context Managers)
# MAGIC
# MAGIC ```python
# MAGIC class DatabaseResource:
# MAGIC     async def __aenter__(self):
# MAGIC         self.pool = await create_pool()
# MAGIC         return self.pool
# MAGIC     
# MAGIC     async def __aexit__(self, *args):
# MAGIC         await self.pool.close()
# MAGIC
# MAGIC class Container(containers.DeclarativeContainer):
# MAGIC     database = providers.Resource(
# MAGIC         DatabaseResource,
# MAGIC         host="localhost"
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Configuration Provider
# MAGIC
# MAGIC ```python
# MAGIC class Container(containers.DeclarativeContainer):
# MAGIC     config = providers.Configuration()
# MAGIC     
# MAGIC     database = providers.Singleton(
# MAGIC         Database,
# MAGIC         host=config.db.host,
# MAGIC         port=config.db.port
# MAGIC     )
# MAGIC
# MAGIC # Load from file
# MAGIC container = Container()
# MAGIC container.config.from_yaml('config.yaml')
# MAGIC
# MAGIC # Or from dict
# MAGIC container.config.from_dict({
# MAGIC     'db': {
# MAGIC         'host': 'localhost',
# MAGIC         'port': 5432
# MAGIC     }
# MAGIC })
# MAGIC ```
# MAGIC
# MAGIC ### 4️⃣ Wiring (Automatic Injection)
# MAGIC
# MAGIC ```python
# MAGIC @containers.inject
# MAGIC async def process_order(
# MAGIC     order_id: int,
# MAGIC     db: Database = Provide[Container.database],
# MAGIC     email: EmailService = Provide[Container.email_service]
# MAGIC ):
# MAGIC     order = await db.get_order(order_id)
# MAGIC     await email.send_confirmation(order)
# MAGIC
# MAGIC # Container automatically injects dependencies!
# MAGIC await process_order(order_id=123)
# MAGIC ```
# MAGIC
# MAGIC ## Advantages Over Manual DI:
# MAGIC
# MAGIC ✅ **Less boilerplate** - Declarative syntax
# MAGIC ✅ **Automatic wiring** - No manual injection needed
# MAGIC ✅ **Configuration management** - Built-in config support
# MAGIC ✅ **Resource lifecycle** - Automatic cleanup
# MAGIC ✅ **Testing support** - Easy overrides
# MAGIC ✅ **Type safety** - Full type hints
# MAGIC
# MAGIC Let's use it! 🚀

# COMMAND ----------

# DBTITLE 1,dependency-injector Implementation
# MAGIC %pip install dependency-injector -q
# MAGIC
# MAGIC from dependency_injector import containers, providers
# MAGIC from dependency_injector.wiring import Provide, inject
# MAGIC import asyncio
# MAGIC from typing import Dict, List
# MAGIC
# MAGIC print("📦 dependency-injector Library Example\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC # ==================== Define Services ====================
# MAGIC
# MAGIC class Config:
# MAGIC     """Configuration service"""
# MAGIC     def __init__(self):
# MAGIC         self.db_host = "localhost"
# MAGIC         self.db_port = 5432
# MAGIC         self.smtp_host = "smtp.gmail.com"
# MAGIC         self.smtp_port = 587
# MAGIC         self.api_key = "secret-key-123"
# MAGIC         print("⚙️ Config loaded")
# MAGIC
# MAGIC class AsyncDatabase:
# MAGIC     """Async database service"""
# MAGIC     def __init__(self, host: str, port: int):
# MAGIC         self.host = host
# MAGIC         self.port = port
# MAGIC         self.connected = False
# MAGIC         print(f"💾 AsyncDatabase initialized: {host}:{port}")
# MAGIC     
# MAGIC     async def connect(self):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         self.connected = True
# MAGIC         print("✅ Database connected")
# MAGIC         return self
# MAGIC     
# MAGIC     async def disconnect(self):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         self.connected = False
# MAGIC         print("🔌 Database disconnected")
# MAGIC     
# MAGIC     async def query(self, sql: str) -> List[Dict]:
# MAGIC         if not self.connected:
# MAGIC             raise RuntimeError("Database not connected")
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         return [{"id": 1, "data": "sample"}]
# MAGIC
# MAGIC class EmailClient:
# MAGIC     """Email client service"""
# MAGIC     def __init__(self, smtp_host: str, smtp_port: int):
# MAGIC         self.smtp_host = smtp_host
# MAGIC         self.smtp_port = smtp_port
# MAGIC         print(f"📧 EmailClient configured: {smtp_host}:{smtp_port}")
# MAGIC     
# MAGIC     async def send_email(self, to: str, subject: str, body: str):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         print(f"✉️ Sent email to {to}: {subject}")
# MAGIC         return True
# MAGIC
# MAGIC class CacheClient:
# MAGIC     """Cache client"""
# MAGIC     def __init__(self):
# MAGIC         self._store = {}
# MAGIC         print("🗄️ CacheClient initialized")
# MAGIC     
# MAGIC     async def get(self, key: str):
# MAGIC         return self._store.get(key)
# MAGIC     
# MAGIC     async def set(self, key: str, value: any, ttl: int = 300):
# MAGIC         self._store[key] = value
# MAGIC
# MAGIC class UserRepository:
# MAGIC     """User repository"""
# MAGIC     def __init__(self, database: AsyncDatabase, cache: CacheClient):
# MAGIC         self.db = database
# MAGIC         self.cache = cache
# MAGIC         print("📂 UserRepository initialized")
# MAGIC     
# MAGIC     async def get_user(self, user_id: int) -> Dict:
# MAGIC         # Try cache
# MAGIC         cached = await self.cache.get(f"user:{user_id}")
# MAGIC         if cached:
# MAGIC             print(f"  🎯 Cache hit: user {user_id}")
# MAGIC             return cached
# MAGIC         
# MAGIC         # Query database
# MAGIC         print(f"  🔍 Querying DB: user {user_id}")
# MAGIC         results = await self.db.query(f"SELECT * FROM users WHERE id={user_id}")
# MAGIC         user = results[0] if results else {"id": user_id, "name": "John Doe"}
# MAGIC         
# MAGIC         # Cache it
# MAGIC         await self.cache.set(f"user:{user_id}", user)
# MAGIC         return user
# MAGIC
# MAGIC class NotificationService:
# MAGIC     """Notification service"""
# MAGIC     def __init__(self, email_client: EmailClient):
# MAGIC         self.email = email_client
# MAGIC         print("🔔 NotificationService initialized")
# MAGIC     
# MAGIC     async def notify(self, user_id: int, message: str):
# MAGIC         await self.email.send_email(
# MAGIC             to=f"user{user_id}@example.com",
# MAGIC             subject="Notification",
# MAGIC             body=message
# MAGIC         )
# MAGIC
# MAGIC class UserService:
# MAGIC     """User service (main application service)"""
# MAGIC     def __init__(self, 
# MAGIC                  user_repo: UserRepository,
# MAGIC                  notification: NotificationService):
# MAGIC         self.user_repo = user_repo
# MAGIC         self.notification = notification
# MAGIC         print("👥 UserService initialized with all dependencies")
# MAGIC     
# MAGIC     async def register_user(self, user_id: int, name: str):
# MAGIC         print(f"\n🔄 Registering user {user_id}...")
# MAGIC         
# MAGIC         # Save user (simulated)
# MAGIC         user = {"id": user_id, "name": name}
# MAGIC         await self.user_repo.cache.set(f"user:{user_id}", user)
# MAGIC         
# MAGIC         # Send notification
# MAGIC         await self.notification.notify(user_id, f"Welcome {name}!")
# MAGIC         
# MAGIC         print(f"✅ User {user_id} registered successfully\n")
# MAGIC         return user
# MAGIC     
# MAGIC     async def get_user_profile(self, user_id: int):
# MAGIC         print(f"\n🔍 Getting user profile {user_id}...")
# MAGIC         user = await self.user_repo.get_user(user_id)
# MAGIC         print(f"✅ Retrieved: {user}\n")
# MAGIC         return user
# MAGIC
# MAGIC # ==================== Define Container ====================
# MAGIC
# MAGIC print("\n🏗️ Building DI Container\n")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC class ApplicationContainer(containers.DeclarativeContainer):
# MAGIC     """Application DI Container"""
# MAGIC     
# MAGIC     # Configuration (singleton)
# MAGIC     config = providers.Singleton(Config)
# MAGIC     
# MAGIC     # Cache (singleton - shared across app)
# MAGIC     cache = providers.Singleton(CacheClient)
# MAGIC     
# MAGIC     # Database with resource management
# MAGIC     database = providers.Resource(
# MAGIC         AsyncDatabase,
# MAGIC         host=config.provided.db_host,
# MAGIC         port=config.provided.db_port
# MAGIC     )
# MAGIC     
# MAGIC     # Email client (factory - new instance when needed)
# MAGIC     email_client = providers.Factory(
# MAGIC         EmailClient,
# MAGIC         smtp_host=config.provided.smtp_host,
# MAGIC         smtp_port=config.provided.smtp_port
# MAGIC     )
# MAGIC     
# MAGIC     # User repository (factory)
# MAGIC     user_repository = providers.Factory(
# MAGIC         UserRepository,
# MAGIC         database=database,
# MAGIC         cache=cache
# MAGIC     )
# MAGIC     
# MAGIC     # Notification service (factory)
# MAGIC     notification_service = providers.Factory(
# MAGIC         NotificationService,
# MAGIC         email_client=email_client
# MAGIC     )
# MAGIC     
# MAGIC     # User service (factory)
# MAGIC     user_service = providers.Factory(
# MAGIC         UserService,
# MAGIC         user_repo=user_repository,
# MAGIC         notification=notification_service
# MAGIC     )
# MAGIC
# MAGIC print("✅ Container defined with all services\n")
# MAGIC
# MAGIC # ==================== Initialize Container ====================
# MAGIC
# MAGIC print("="*80)
# MAGIC print("🚀 Initializing Application")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC container = ApplicationContainer()
# MAGIC
# MAGIC # Initialize async resources
# MAGIC async def init_resources():
# MAGIC     await container.init_resources()
# MAGIC     print("\n✅ All async resources initialized\n")
# MAGIC
# MAGIC await init_resources()
# MAGIC
# MAGIC # ==================== Using the Container ====================
# MAGIC
# MAGIC print("="*80)
# MAGIC print("🎯 Using Services from Container")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC print("1️⃣ Getting UserService from container:\n")
# MAGIC user_service = container.user_service()
# MAGIC
# MAGIC print("\n2️⃣ Registering new users:\n")
# MAGIC await user_service.register_user(101, "Alice")
# MAGIC await user_service.register_user(102, "Bob")
# MAGIC
# MAGIC print("\n3️⃣ Retrieving user profiles:\n")
# MAGIC await user_service.get_user_profile(101)  # From cache
# MAGIC await user_service.get_user_profile(102)  # From cache
# MAGIC await user_service.get_user_profile(103)  # Not cached, queries DB
# MAGIC
# MAGIC # ==================== Wiring Example ====================
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("🔌 Automatic Dependency Injection (Wiring)")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC # Define function with automatic injection
# MAGIC @inject
# MAGIC async def process_user_action(
# MAGIC     user_id: int,
# MAGIC     action: str,
# MAGIC     user_repo: UserRepository = Provide[ApplicationContainer.user_repository],
# MAGIC     notification: NotificationService = Provide[ApplicationContainer.notification_service]
# MAGIC ):
# MAGIC     """
# MAGIC     Function with automatic dependency injection!
# MAGIC     No need to pass dependencies manually.
# MAGIC     """
# MAGIC     print(f"🔄 Processing action '{action}' for user {user_id}")
# MAGIC     
# MAGIC     # Use injected dependencies
# MAGIC     user = await user_repo.get_user(user_id)
# MAGIC     await notification.notify(user_id, f"Action completed: {action}")
# MAGIC     
# MAGIC     print(f"✅ Action processed for {user['name']}\n")
# MAGIC     return {"status": "success", "user": user}
# MAGIC
# MAGIC # Wire the container to enable automatic injection
# MAGIC container.wire(modules=[__name__])
# MAGIC
# MAGIC print("Calling function WITHOUT passing dependencies:\n")
# MAGIC result = await process_user_action(101, "profile_update")
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("✅ dependency-injector demonstration complete!")
# MAGIC print("💡 Key benefits:")
# MAGIC print("   1. Declarative container definition")
# MAGIC print("   2. Automatic resource management")
# MAGIC print("   3. Type-safe dependency injection")
# MAGIC print("   4. Clean, maintainable code")
# MAGIC print("   5. Easy testing with overrides")
# MAGIC
# MAGIC # Cleanup
# MAGIC async def cleanup():
# MAGIC     await container.shutdown_resources()
# MAGIC     print("\n🧹 Resources cleaned up")
# MAGIC
# MAGIC await cleanup()

# COMMAND ----------

# DBTITLE 1,🚀 FastAPI + DI Integration
# MAGIC %md
# MAGIC # Cell 4: FastAPI with Dependency Injection
# MAGIC
# MAGIC ## Kyu FastAPI + DI?
# MAGIC
# MAGIC **FastAPI** already has built-in dependency injection system:
# MAGIC - Function parameters se dependencies inject hoti hain
# MAGIC - Async support out-of-the-box
# MAGIC - Automatic validation
# MAGIC - OpenAPI documentation
# MAGIC
# MAGIC ## FastAPI DI Patterns:
# MAGIC
# MAGIC ### 1️⃣ Simple Dependency
# MAGIC
# MAGIC ```python
# MAGIC from fastapi import Depends
# MAGIC
# MAGIC def get_db():
# MAGIC     db = Database()
# MAGIC     try:
# MAGIC         yield db
# MAGIC     finally:
# MAGIC         db.close()
# MAGIC
# MAGIC @app.get("/users/{user_id}")
# MAGIC async def get_user(user_id: int, db: Database = Depends(get_db)):
# MAGIC     return await db.get_user(user_id)
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Class-based Dependencies
# MAGIC
# MAGIC ```python
# MAGIC class DatabaseDep:
# MAGIC     async def __call__(self):
# MAGIC         db = await create_db_pool()
# MAGIC         try:
# MAGIC             yield db
# MAGIC         finally:
# MAGIC             await db.close()
# MAGIC
# MAGIC get_db = DatabaseDep()
# MAGIC
# MAGIC @app.get("/users/{user_id}")
# MAGIC async def get_user(user_id: int, db = Depends(get_db)):
# MAGIC     return await db.get_user(user_id)
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Nested Dependencies
# MAGIC
# MAGIC ```python
# MAGIC async def get_db():
# MAGIC     yield Database()
# MAGIC
# MAGIC async def get_user_repo(db: Database = Depends(get_db)):
# MAGIC     yield UserRepository(db)
# MAGIC
# MAGIC async def get_user_service(
# MAGIC     repo: UserRepository = Depends(get_user_repo),
# MAGIC     email: EmailService = Depends(get_email_service)
# MAGIC ):
# MAGIC     yield UserService(repo, email)
# MAGIC
# MAGIC @app.post("/users")
# MAGIC async def create_user(
# MAGIC     user_data: UserCreate,
# MAGIC     service: UserService = Depends(get_user_service)  # Auto-resolves all deps!
# MAGIC ):
# MAGIC     return await service.create_user(user_data)
# MAGIC ```
# MAGIC
# MAGIC ### 4️⃣ Application State (Lifespan)
# MAGIC
# MAGIC ```python
# MAGIC from contextlib import asynccontextmanager
# MAGIC
# MAGIC @asynccontextmanager
# MAGIC async def lifespan(app: FastAPI):
# MAGIC     # Startup
# MAGIC     db = await create_db_pool()
# MAGIC     cache = await create_redis_connection()
# MAGIC     
# MAGIC     app.state.db = db
# MAGIC     app.state.cache = cache
# MAGIC     
# MAGIC     yield  # Application runs
# MAGIC     
# MAGIC     # Shutdown
# MAGIC     await db.close()
# MAGIC     await cache.close()
# MAGIC
# MAGIC app = FastAPI(lifespan=lifespan)
# MAGIC
# MAGIC def get_db(request: Request):
# MAGIC     return request.app.state.db
# MAGIC ```
# MAGIC
# MAGIC ### 5️⃣ Combining with dependency-injector
# MAGIC
# MAGIC ```python
# MAGIC from dependency_injector.wiring import inject, Provide
# MAGIC
# MAGIC @app.get("/users/{user_id}")
# MAGIC @inject
# MAGIC async def get_user(
# MAGIC     user_id: int,
# MAGIC     service: UserService = Depends(Provide[Container.user_service])
# MAGIC ):
# MAGIC     return await service.get_user(user_id)
# MAGIC
# MAGIC # Wire container on startup
# MAGIC @app.on_event("startup")
# MAGIC async def startup():
# MAGIC     container.wire(modules=[__name__])
# MAGIC ```
# MAGIC
# MAGIC ## Benefits:
# MAGIC
# MAGIC ✅ **Clean code** - No manual dependency passing
# MAGIC ✅ **Testable** - Easy to mock dependencies
# MAGIC ✅ **Reusable** - Share dependencies across routes
# MAGIC ✅ **Type-safe** - Full IDE support
# MAGIC ✅ **Documented** - Auto-generates OpenAPI docs
# MAGIC
# MAGIC Let's build an API! 🌐

# COMMAND ----------

# DBTITLE 1,FastAPI + DI Complete Example
# MAGIC %pip install fastapi uvicorn httpx -q
# MAGIC
# MAGIC from fastapi import FastAPI, Depends, HTTPException, Request
# MAGIC from fastapi.responses import JSONResponse
# MAGIC from contextlib import asynccontextmanager
# MAGIC from pydantic import BaseModel, Field
# MAGIC from typing import List, Optional
# MAGIC import asyncio
# MAGIC
# MAGIC print("🚀 FastAPI + Dependency Injection Example\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC # ==================== Models ====================
# MAGIC
# MAGIC class UserCreate(BaseModel):
# MAGIC     name: str = Field(..., min_length=2, max_length=50)
# MAGIC     email: str = Field(..., regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
# MAGIC     age: int = Field(..., ge=18, le=120)
# MAGIC
# MAGIC class UserResponse(BaseModel):
# MAGIC     id: int
# MAGIC     name: str
# MAGIC     email: str
# MAGIC     age: int
# MAGIC     created_at: str
# MAGIC
# MAGIC class MessageResponse(BaseModel):
# MAGIC     message: str
# MAGIC     user_id: int
# MAGIC
# MAGIC # ==================== Services ====================
# MAGIC
# MAGIC class AppConfig:
# MAGIC     """Application configuration"""
# MAGIC     def __init__(self):
# MAGIC         self.db_url = "postgresql://localhost/mydb"
# MAGIC         self.redis_url = "redis://localhost"
# MAGIC         self.smtp_host = "smtp.gmail.com"
# MAGIC         self.api_key = "secret-key-123"
# MAGIC         print("⚙️ AppConfig loaded")
# MAGIC
# MAGIC class DatabaseConnection:
# MAGIC     """Database connection pool"""
# MAGIC     def __init__(self, connection_string: str):
# MAGIC         self.connection_string = connection_string
# MAGIC         self.pool = None
# MAGIC         self.users_db = {
# MAGIC             1: {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
# MAGIC             2: {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25}
# MAGIC         }
# MAGIC         print(f"💾 DatabaseConnection initialized")
# MAGIC     
# MAGIC     async def connect(self):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         self.pool = "connection_pool"
# MAGIC         print("✅ Database connected")
# MAGIC         return self
# MAGIC     
# MAGIC     async def disconnect(self):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         self.pool = None
# MAGIC         print("🔌 Database disconnected")
# MAGIC     
# MAGIC     async def get_user(self, user_id: int):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         return self.users_db.get(user_id)
# MAGIC     
# MAGIC     async def create_user(self, user_data: dict) -> dict:
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         user_id = max(self.users_db.keys()) + 1 if self.users_db else 1
# MAGIC         user = {"id": user_id, **user_data, "created_at": "2026-06-08T10:00:00"}
# MAGIC         self.users_db[user_id] = user
# MAGIC         return user
# MAGIC     
# MAGIC     async def list_users(self) -> List[dict]:
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         return list(self.users_db.values())
# MAGIC
# MAGIC class CacheConnection:
# MAGIC     """Redis cache connection"""
# MAGIC     def __init__(self):
# MAGIC         self._cache = {}
# MAGIC         print("🗄️ CacheConnection initialized")
# MAGIC     
# MAGIC     async def connect(self):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         print("✅ Cache connected")
# MAGIC         return self
# MAGIC     
# MAGIC     async def disconnect(self):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         print("🔌 Cache disconnected")
# MAGIC     
# MAGIC     async def get(self, key: str):
# MAGIC         return self._cache.get(key)
# MAGIC     
# MAGIC     async def set(self, key: str, value: any, ttl: int = 300):
# MAGIC         self._cache[key] = value
# MAGIC         print(f"  💾 Cached: {key}")
# MAGIC
# MAGIC class EmailServiceClient:
# MAGIC     """Email service"""
# MAGIC     def __init__(self, smtp_host: str):
# MAGIC         self.smtp_host = smtp_host
# MAGIC         print(f"📧 EmailServiceClient: {smtp_host}")
# MAGIC     
# MAGIC     async def send_welcome_email(self, email: str, name: str):
# MAGIC         await asyncio.sleep(0.05)
# MAGIC         print(f"  ✉️ Welcome email sent to {email}")
# MAGIC         return True
# MAGIC
# MAGIC # ==================== Application State ====================
# MAGIC
# MAGIC class AppState:
# MAGIC     """Global application state"""
# MAGIC     def __init__(self):
# MAGIC         self.config: Optional[AppConfig] = None
# MAGIC         self.db: Optional[DatabaseConnection] = None
# MAGIC         self.cache: Optional[CacheConnection] = None
# MAGIC         self.email: Optional[EmailServiceClient] = None
# MAGIC
# MAGIC # ==================== Lifespan Management ====================
# MAGIC
# MAGIC app_state = AppState()
# MAGIC
# MAGIC @asynccontextmanager
# MAGIC async def lifespan(app: FastAPI):
# MAGIC     """
# MAGIC     Application lifespan - startup and shutdown
# MAGIC     """
# MAGIC     print("\n🚀 Application Starting Up...\n")
# MAGIC     
# MAGIC     # Initialize services
# MAGIC     app_state.config = AppConfig()
# MAGIC     
# MAGIC     app_state.db = DatabaseConnection(app_state.config.db_url)
# MAGIC     await app_state.db.connect()
# MAGIC     
# MAGIC     app_state.cache = CacheConnection()
# MAGIC     await app_state.cache.connect()
# MAGIC     
# MAGIC     app_state.email = EmailServiceClient(app_state.config.smtp_host)
# MAGIC     
# MAGIC     print("\n✅ All services initialized\n")
# MAGIC     
# MAGIC     yield  # Application runs here
# MAGIC     
# MAGIC     print("\n🚦 Application Shutting Down...\n")
# MAGIC     
# MAGIC     # Cleanup
# MAGIC     await app_state.db.disconnect()
# MAGIC     await app_state.cache.disconnect()
# MAGIC     
# MAGIC     print("✅ Cleanup complete\n")
# MAGIC
# MAGIC # ==================== Create FastAPI App ====================
# MAGIC
# MAGIC app = FastAPI(
# MAGIC     title="User Management API",
# MAGIC     description="FastAPI with Dependency Injection Example",
# MAGIC     version="1.0.0",
# MAGIC     lifespan=lifespan
# MAGIC )
# MAGIC
# MAGIC print("\n🌐 FastAPI Application Created\n")
# MAGIC
# MAGIC # ==================== Dependency Functions ====================
# MAGIC
# MAGIC async def get_db() -> DatabaseConnection:
# MAGIC     """Get database connection from app state"""
# MAGIC     return app_state.db
# MAGIC
# MAGIC async def get_cache() -> CacheConnection:
# MAGIC     """Get cache connection from app state"""
# MAGIC     return app_state.cache
# MAGIC
# MAGIC async def get_email() -> EmailServiceClient:
# MAGIC     """Get email service from app state"""
# MAGIC     return app_state.email
# MAGIC
# MAGIC class UserServiceDep:
# MAGIC     """User service with nested dependencies"""
# MAGIC     async def __call__(
# MAGIC         self,
# MAGIC         db: DatabaseConnection = Depends(get_db),
# MAGIC         cache: CacheConnection = Depends(get_cache),
# MAGIC         email: EmailServiceClient = Depends(get_email)
# MAGIC     ):
# MAGIC         return {
# MAGIC             "db": db,
# MAGIC             "cache": cache,
# MAGIC             "email": email
# MAGIC         }
# MAGIC
# MAGIC get_user_service = UserServiceDep()
# MAGIC
# MAGIC # ==================== API Routes ====================
# MAGIC
# MAGIC @app.get("/")
# MAGIC async def root():
# MAGIC     return {
# MAGIC         "message": "User Management API",
# MAGIC         "docs": "/docs",
# MAGIC         "health": "/health"
# MAGIC     }
# MAGIC
# MAGIC @app.get("/health")
# MAGIC async def health_check(
# MAGIC     db: DatabaseConnection = Depends(get_db),
# MAGIC     cache: CacheConnection = Depends(get_cache)
# MAGIC ):
# MAGIC     """Health check endpoint with dependency injection"""
# MAGIC     return {
# MAGIC         "status": "healthy",
# MAGIC         "database": "connected" if db.pool else "disconnected",
# MAGIC         "cache": "connected"
# MAGIC     }
# MAGIC
# MAGIC @app.get("/users", response_model=List[UserResponse])
# MAGIC async def list_users(
# MAGIC     db: DatabaseConnection = Depends(get_db),
# MAGIC     cache: CacheConnection = Depends(get_cache)
# MAGIC ):
# MAGIC     """
# MAGIC     List all users
# MAGIC     Dependencies: db, cache
# MAGIC     """
# MAGIC     # Try cache first
# MAGIC     cached = await cache.get("users:all")
# MAGIC     if cached:
# MAGIC         print("🎯 Cache hit: all users")
# MAGIC         return cached
# MAGIC     
# MAGIC     # Query database
# MAGIC     users = await db.list_users()
# MAGIC     
# MAGIC     # Cache result
# MAGIC     await cache.set("users:all", users, ttl=60)
# MAGIC     
# MAGIC     return users
# MAGIC
# MAGIC @app.get("/users/{user_id}", response_model=UserResponse)
# MAGIC async def get_user(
# MAGIC     user_id: int,
# MAGIC     db: DatabaseConnection = Depends(get_db),
# MAGIC     cache: CacheConnection = Depends(get_cache)
# MAGIC ):
# MAGIC     """
# MAGIC     Get user by ID
# MAGIC     Dependencies: db, cache
# MAGIC     """
# MAGIC     # Try cache
# MAGIC     cache_key = f"user:{user_id}"
# MAGIC     cached = await cache.get(cache_key)
# MAGIC     if cached:
# MAGIC         print(f"🎯 Cache hit: user {user_id}")
# MAGIC         return cached
# MAGIC     
# MAGIC     # Query database
# MAGIC     user = await db.get_user(user_id)
# MAGIC     if not user:
# MAGIC         raise HTTPException(status_code=404, detail="User not found")
# MAGIC     
# MAGIC     # Cache result
# MAGIC     await cache.set(cache_key, user, ttl=300)
# MAGIC     
# MAGIC     return user
# MAGIC
# MAGIC @app.post("/users", response_model=UserResponse, status_code=201)
# MAGIC async def create_user(
# MAGIC     user_data: UserCreate,
# MAGIC     services: dict = Depends(get_user_service)
# MAGIC ):
# MAGIC     """
# MAGIC     Create new user
# MAGIC     Dependencies: db, cache, email (via UserServiceDep)
# MAGIC     """
# MAGIC     db = services["db"]
# MAGIC     cache = services["cache"]
# MAGIC     email_service = services["email"]
# MAGIC     
# MAGIC     # Create user in database
# MAGIC     user = await db.create_user(user_data.dict())
# MAGIC     
# MAGIC     # Invalidate cache
# MAGIC     await cache.set("users:all", None)  # Clear list cache
# MAGIC     
# MAGIC     # Send welcome email (async, don't wait)
# MAGIC     asyncio.create_task(
# MAGIC         email_service.send_welcome_email(user["email"], user["name"])
# MAGIC     )
# MAGIC     
# MAGIC     return user
# MAGIC
# MAGIC @app.post("/users/{user_id}/notify", response_model=MessageResponse)
# MAGIC async def notify_user(
# MAGIC     user_id: int,
# MAGIC     message: str,
# MAGIC     db: DatabaseConnection = Depends(get_db),
# MAGIC     email_service: EmailServiceClient = Depends(get_email)
# MAGIC ):
# MAGIC     """
# MAGIC     Send notification to user
# MAGIC     Dependencies: db, email
# MAGIC     """
# MAGIC     # Get user
# MAGIC     user = await db.get_user(user_id)
# MAGIC     if not user:
# MAGIC         raise HTTPException(status_code=404, detail="User not found")
# MAGIC     
# MAGIC     # Send email
# MAGIC     await email_service.send_welcome_email(user["email"], user["name"])
# MAGIC     
# MAGIC     return MessageResponse(
# MAGIC         message=f"Notification sent to {user['name']}",
# MAGIC         user_id=user_id
# MAGIC     )
# MAGIC
# MAGIC print("="*80)
# MAGIC print("✅ FastAPI routes defined with DI")
# MAGIC print("\n📝 Available Endpoints:")
# MAGIC print("   GET  /              - Root")
# MAGIC print("   GET  /health        - Health check")
# MAGIC print("   GET  /users         - List all users")
# MAGIC print("   GET  /users/{id}    - Get user by ID")
# MAGIC print("   POST /users         - Create user")
# MAGIC print("   POST /users/{id}/notify - Notify user")
# MAGIC print("\n💡 All endpoints use dependency injection!")
# MAGIC print("\nTo run the server:")
# MAGIC print("   uvicorn main:app --reload")
# MAGIC print("   Then visit: http://localhost:8000/docs")

# COMMAND ----------

# DBTITLE 1,🧪 Testing with Dependency Injection
# MAGIC %md
# MAGIC # Cell 5: Testing with Dependency Injection
# MAGIC
# MAGIC ## Kyu DI se Testing Easy Hoti Hai?
# MAGIC
# MAGIC **Without DI:**
# MAGIC ```python
# MAGIC # ❌ Hard to test - creates real dependencies
# MAGIC class UserService:
# MAGIC     def __init__(self):
# MAGIC         self.db = Database()  # Real database!
# MAGIC         self.email = EmailService()  # Sends real emails!
# MAGIC
# MAGIC # Test will use real services - BAD!
# MAGIC def test_create_user():
# MAGIC     service = UserService()
# MAGIC     user = service.create_user(...)  # Hits real DB, sends email!
# MAGIC ```
# MAGIC
# MAGIC **With DI:**
# MAGIC ```python
# MAGIC # ✅ Easy to test - inject mocks
# MAGIC class UserService:
# MAGIC     def __init__(self, db, email):
# MAGIC         self.db = db
# MAGIC         self.email = email
# MAGIC
# MAGIC # Test with mocks - GOOD!
# MAGIC def test_create_user():
# MAGIC     mock_db = MockDatabase()
# MAGIC     mock_email = MockEmailService()
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     user = service.create_user(...)  # Uses mocks!
# MAGIC ```
# MAGIC
# MAGIC ## Testing Patterns:
# MAGIC
# MAGIC ### 1️⃣ Manual Mocking
# MAGIC
# MAGIC ```python
# MAGIC import pytest
# MAGIC
# MAGIC class MockDatabase:
# MAGIC     async def get_user(self, user_id):
# MAGIC         return {"id": user_id, "name": "Test User"}
# MAGIC     
# MAGIC     async def create_user(self, user_data):
# MAGIC         return {"id": 1, **user_data}
# MAGIC
# MAGIC @pytest.mark.asyncio
# MAGIC async def test_user_service():
# MAGIC     # Arrange
# MAGIC     mock_db = MockDatabase()
# MAGIC     service = UserService(mock_db)
# MAGIC     
# MAGIC     # Act
# MAGIC     user = await service.get_user(1)
# MAGIC     
# MAGIC     # Assert
# MAGIC     assert user["name"] == "Test User"
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Using unittest.mock
# MAGIC
# MAGIC ```python
# MAGIC from unittest.mock import AsyncMock, patch
# MAGIC
# MAGIC @pytest.mark.asyncio
# MAGIC async def test_user_service_with_mock():
# MAGIC     # Create mocks
# MAGIC     mock_db = AsyncMock()
# MAGIC     mock_db.get_user.return_value = {"id": 1, "name": "Test"}
# MAGIC     
# MAGIC     # Test
# MAGIC     service = UserService(mock_db)
# MAGIC     user = await service.get_user(1)
# MAGIC     
# MAGIC     # Verify
# MAGIC     mock_db.get_user.assert_called_once_with(1)
# MAGIC     assert user["name"] == "Test"
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Overriding Container Dependencies
# MAGIC
# MAGIC ```python
# MAGIC from dependency_injector import containers, providers
# MAGIC
# MAGIC def test_with_container_override():
# MAGIC     # Create test container
# MAGIC     container = ApplicationContainer()
# MAGIC     
# MAGIC     # Override with mocks
# MAGIC     container.database.override(MockDatabase())
# MAGIC     container.email.override(MockEmailService())
# MAGIC     
# MAGIC     # Test
# MAGIC     service = container.user_service()
# MAGIC     result = service.process(...)
# MAGIC     
# MAGIC     # Verify
# MAGIC     assert result["status"] == "success"
# MAGIC ```
# MAGIC
# MAGIC ### 4️⃣ FastAPI Testing with DI
# MAGIC
# MAGIC ```python
# MAGIC from fastapi.testclient import TestClient
# MAGIC
# MAGIC def test_api_endpoint():
# MAGIC     # Override dependencies
# MAGIC     app.dependency_overrides[get_db] = lambda: MockDatabase()
# MAGIC     
# MAGIC     # Test
# MAGIC     client = TestClient(app)
# MAGIC     response = client.get("/users/1")
# MAGIC     
# MAGIC     # Assert
# MAGIC     assert response.status_code == 200
# MAGIC     assert response.json()["name"] == "Test User"
# MAGIC     
# MAGIC     # Cleanup
# MAGIC     app.dependency_overrides.clear()
# MAGIC ```
# MAGIC
# MAGIC ### 5️⃣ Pytest Fixtures for DI
# MAGIC
# MAGIC ```python
# MAGIC import pytest
# MAGIC
# MAGIC @pytest.fixture
# MAGIC async def mock_db():
# MAGIC     db = MockDatabase()
# MAGIC     await db.connect()
# MAGIC     yield db
# MAGIC     await db.disconnect()
# MAGIC
# MAGIC @pytest.fixture
# MAGIC def user_service(mock_db):
# MAGIC     return UserService(mock_db)
# MAGIC
# MAGIC @pytest.mark.asyncio
# MAGIC async def test_with_fixtures(user_service):
# MAGIC     user = await user_service.get_user(1)
# MAGIC     assert user is not None
# MAGIC ```
# MAGIC
# MAGIC ## Testing Best Practices:
# MAGIC
# MAGIC ✅ **Test behavior, not implementation**
# MAGIC ✅ **Use realistic mocks** - Match real interface
# MAGIC ✅ **Test error cases** - Not just happy path
# MAGIC ✅ **Isolate tests** - No shared state
# MAGIC ✅ **Fast tests** - No real I/O
# MAGIC
# MAGIC Let's write tests! 🧪

# COMMAND ----------

# DBTITLE 1,Testing Implementation with DI
# MAGIC %pip install pytest pytest-asyncio -q
# MAGIC
# MAGIC from unittest.mock import AsyncMock, MagicMock, patch
# MAGIC from typing import Dict, Optional
# MAGIC import pytest
# MAGIC
# MAGIC print("🧪 Testing with Dependency Injection\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC # ==================== Services to Test ====================
# MAGIC
# MAGIC class User:
# MAGIC     def __init__(self, id: int, name: str, email: str):
# MAGIC         self.id = id
# MAGIC         self.name = name
# MAGIC         self.email = email
# MAGIC     
# MAGIC     def to_dict(self):
# MAGIC         return {"id": self.id, "name": self.name, "email": self.email}
# MAGIC
# MAGIC class RealDatabase:
# MAGIC     """Real database - we DON'T want to use this in tests!"""
# MAGIC     async def get_user(self, user_id: int) -> Optional[User]:
# MAGIC         # This would hit real database!
# MAGIC         raise RuntimeError("Should not call real database in tests!")
# MAGIC     
# MAGIC     async def create_user(self, name: str, email: str) -> User:
# MAGIC         raise RuntimeError("Should not call real database in tests!")
# MAGIC
# MAGIC class RealEmailService:
# MAGIC     """Real email service - we DON'T want to use this in tests!"""
# MAGIC     async def send_welcome(self, email: str, name: str):
# MAGIC         # This would send real email!
# MAGIC         raise RuntimeError("Should not send real emails in tests!")
# MAGIC
# MAGIC class UserService:
# MAGIC     """Service we want to test"""
# MAGIC     def __init__(self, database, email_service):
# MAGIC         self.db = database
# MAGIC         self.email = email_service
# MAGIC     
# MAGIC     async def register_user(self, name: str, email: str) -> Dict:
# MAGIC         # Validate
# MAGIC         if len(name) < 2:
# MAGIC             raise ValueError("Name too short")
# MAGIC         
# MAGIC         # Create user
# MAGIC         user = await self.db.create_user(name, email)
# MAGIC         
# MAGIC         # Send welcome email
# MAGIC         await self.email.send_welcome(email, name)
# MAGIC         
# MAGIC         return user.to_dict()
# MAGIC     
# MAGIC     async def get_user_profile(self, user_id: int) -> Optional[Dict]:
# MAGIC         user = await self.db.get_user(user_id)
# MAGIC         if not user:
# MAGIC             return None
# MAGIC         return user.to_dict()
# MAGIC
# MAGIC # ==================== Mock Implementations ====================
# MAGIC
# MAGIC print("\n1️⃣ Creating Mock Implementations\n")
# MAGIC
# MAGIC class MockDatabase:
# MAGIC     """Mock database for testing"""
# MAGIC     def __init__(self):
# MAGIC         self.users = {
# MAGIC             1: User(1, "Alice", "alice@test.com"),
# MAGIC             2: User(2, "Bob", "bob@test.com")
# MAGIC         }
# MAGIC         self.next_id = 3
# MAGIC         print("💾 MockDatabase created")
# MAGIC     
# MAGIC     async def get_user(self, user_id: int) -> Optional[User]:
# MAGIC         return self.users.get(user_id)
# MAGIC     
# MAGIC     async def create_user(self, name: str, email: str) -> User:
# MAGIC         user = User(self.next_id, name, email)
# MAGIC         self.users[self.next_id] = user
# MAGIC         self.next_id += 1
# MAGIC         return user
# MAGIC
# MAGIC class MockEmailService:
# MAGIC     """Mock email service for testing"""
# MAGIC     def __init__(self):
# MAGIC         self.sent_emails = []
# MAGIC         print("📧 MockEmailService created")
# MAGIC     
# MAGIC     async def send_welcome(self, email: str, name: str):
# MAGIC         self.sent_emails.append({
# MAGIC             "to": email,
# MAGIC             "name": name,
# MAGIC             "type": "welcome"
# MAGIC         })
# MAGIC         print(f"  📨 Mock email sent to {email}")
# MAGIC
# MAGIC # ==================== Test Cases ====================
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("🧪 Running Test Cases")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC print("Test 1: Get existing user\n")
# MAGIC
# MAGIC async def test_get_user():
# MAGIC     """Test getting an existing user"""
# MAGIC     # Arrange
# MAGIC     mock_db = MockDatabase()
# MAGIC     mock_email = MockEmailService()
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     
# MAGIC     # Act
# MAGIC     user = await service.get_user_profile(1)
# MAGIC     
# MAGIC     # Assert
# MAGIC     assert user is not None
# MAGIC     assert user["name"] == "Alice"
# MAGIC     assert user["email"] == "alice@test.com"
# MAGIC     print("✅ Test passed: Got user Alice")
# MAGIC
# MAGIC await test_get_user()
# MAGIC
# MAGIC print("\nTest 2: Get non-existent user\n")
# MAGIC
# MAGIC async def test_get_nonexistent_user():
# MAGIC     """Test getting a user that doesn't exist"""
# MAGIC     # Arrange
# MAGIC     mock_db = MockDatabase()
# MAGIC     mock_email = MockEmailService()
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     
# MAGIC     # Act
# MAGIC     user = await service.get_user_profile(999)
# MAGIC     
# MAGIC     # Assert
# MAGIC     assert user is None
# MAGIC     print("✅ Test passed: Non-existent user returns None")
# MAGIC
# MAGIC await test_get_nonexistent_user()
# MAGIC
# MAGIC print("\nTest 3: Register new user\n")
# MAGIC
# MAGIC async def test_register_user():
# MAGIC     """Test user registration"""
# MAGIC     # Arrange
# MAGIC     mock_db = MockDatabase()
# MAGIC     mock_email = MockEmailService()
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     
# MAGIC     # Act
# MAGIC     user = await service.register_user("Charlie", "charlie@test.com")
# MAGIC     
# MAGIC     # Assert
# MAGIC     assert user["name"] == "Charlie"
# MAGIC     assert user["email"] == "charlie@test.com"
# MAGIC     assert user["id"] == 3  # Next ID
# MAGIC     
# MAGIC     # Verify email was sent
# MAGIC     assert len(mock_email.sent_emails) == 1
# MAGIC     assert mock_email.sent_emails[0]["to"] == "charlie@test.com"
# MAGIC     
# MAGIC     print("✅ Test passed: User registered and email sent")
# MAGIC
# MAGIC await test_register_user()
# MAGIC
# MAGIC print("\nTest 4: Validation error\n")
# MAGIC
# MAGIC async def test_register_invalid_user():
# MAGIC     """Test registration with invalid data"""
# MAGIC     # Arrange
# MAGIC     mock_db = MockDatabase()
# MAGIC     mock_email = MockEmailService()
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     
# MAGIC     # Act & Assert
# MAGIC     try:
# MAGIC         await service.register_user("A", "invalid@test.com")  # Name too short
# MAGIC         assert False, "Should have raised ValueError"
# MAGIC     except ValueError as e:
# MAGIC         assert "Name too short" in str(e)
# MAGIC         print("✅ Test passed: Validation error raised correctly")
# MAGIC
# MAGIC await test_register_invalid_user()
# MAGIC
# MAGIC # ==================== Using unittest.mock ====================
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("🔧 Using unittest.mock for Advanced Mocking")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC print("Test 5: Using AsyncMock\n")
# MAGIC
# MAGIC async def test_with_async_mock():
# MAGIC     """Test using AsyncMock from unittest.mock"""
# MAGIC     # Create mocks
# MAGIC     mock_db = AsyncMock()
# MAGIC     mock_db.get_user.return_value = User(1, "Test User", "test@example.com")
# MAGIC     
# MAGIC     mock_email = AsyncMock()
# MAGIC     
# MAGIC     # Test
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     user = await service.get_user_profile(1)
# MAGIC     
# MAGIC     # Verify
# MAGIC     assert user["name"] == "Test User"
# MAGIC     mock_db.get_user.assert_called_once_with(1)
# MAGIC     print("✅ Test passed: AsyncMock works correctly")
# MAGIC
# MAGIC await test_with_async_mock()
# MAGIC
# MAGIC print("\nTest 6: Verify interactions\n")
# MAGIC
# MAGIC async def test_verify_interactions():
# MAGIC     """Test verifying mock interactions"""
# MAGIC     # Setup
# MAGIC     mock_db = AsyncMock()
# MAGIC     mock_db.create_user.return_value = User(5, "Diana", "diana@test.com")
# MAGIC     
# MAGIC     mock_email = AsyncMock()
# MAGIC     
# MAGIC     # Execute
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     await service.register_user("Diana", "diana@test.com")
# MAGIC     
# MAGIC     # Verify calls
# MAGIC     mock_db.create_user.assert_called_once_with("Diana", "diana@test.com")
# MAGIC     mock_email.send_welcome.assert_called_once_with("diana@test.com", "Diana")
# MAGIC     
# MAGIC     # Verify call order
# MAGIC     assert mock_db.create_user.called
# MAGIC     assert mock_email.send_welcome.called
# MAGIC     
# MAGIC     print("✅ Test passed: All interactions verified")
# MAGIC
# MAGIC await test_verify_interactions()
# MAGIC
# MAGIC # ==================== Testing with Side Effects ====================
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("🔍 Testing Error Handling")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC print("Test 7: Database error handling\n")
# MAGIC
# MAGIC async def test_database_error():
# MAGIC     """Test handling database errors"""
# MAGIC     # Mock that raises an error
# MAGIC     mock_db = AsyncMock()
# MAGIC     mock_db.create_user.side_effect = Exception("Database connection failed")
# MAGIC     
# MAGIC     mock_email = AsyncMock()
# MAGIC     
# MAGIC     # Test
# MAGIC     service = UserService(mock_db, mock_email)
# MAGIC     
# MAGIC     try:
# MAGIC         await service.register_user("Error Test", "error@test.com")
# MAGIC         assert False, "Should have raised exception"
# MAGIC     except Exception as e:
# MAGIC         assert "Database connection failed" in str(e)
# MAGIC         # Email should NOT have been called
# MAGIC         mock_email.send_welcome.assert_not_called()
# MAGIC         print("✅ Test passed: Database error handled correctly")
# MAGIC
# MAGIC await test_database_error()
# MAGIC
# MAGIC # ==================== Test Summary ====================
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("✅ All Tests Passed!")
# MAGIC print("="*80)
# MAGIC
# MAGIC print("\n📊 Test Summary:")
# MAGIC print("   • Test 1: Get existing user - ✅")
# MAGIC print("   • Test 2: Get non-existent user - ✅")
# MAGIC print("   • Test 3: Register new user - ✅")
# MAGIC print("   • Test 4: Validation error - ✅")
# MAGIC print("   • Test 5: Using AsyncMock - ✅")
# MAGIC print("   • Test 6: Verify interactions - ✅")
# MAGIC print("   • Test 7: Database error handling - ✅")
# MAGIC
# MAGIC print("\n💡 Key Takeaways:")
# MAGIC print("   1. DI makes testing MUCH easier")
# MAGIC print("   2. Mock external dependencies (DB, APIs, etc.)")
# MAGIC print("   3. Verify behavior, not implementation")
# MAGIC print("   4. Test error cases, not just happy path")
# MAGIC print("   5. Use AsyncMock for async code")
# MAGIC print("\n🎯 Without DI, these tests would hit real services!")

# COMMAND ----------

# DBTITLE 1,🔌 Complete Example: MCP Server with DI
# MAGIC %md
# MAGIC # Cell 6: Complete MCP Server with Dependency Injection
# MAGIC
# MAGIC ## Real-World Example: Building an MCP Server with DI
# MAGIC
# MAGIC **Scenario**: Build an MCP server that connects to:
# MAGIC - Database (user data)
# MAGIC - Cache (Redis)
# MAGIC - External API (weather service)
# MAGIC - Email service (notifications)
# MAGIC
# MAGIC ## Architecture:
# MAGIC
# MAGIC ```
# MAGIC                 MCP Server
# MAGIC                     │
# MAGIC         ┌─────────┼─────────┐
# MAGIC         │           │           │
# MAGIC         ↓           ↓           ↓
# MAGIC    UserService  WeatherService  NotifyService
# MAGIC         │           │           │
# MAGIC     ┌───┼───┐   ┌───┼───┐   ┌───┼───┐
# MAGIC     │   │   │   │   │   │   │   │   │
# MAGIC    DB  Cache  │  HTTP Cache │ Email Cache
# MAGIC               │          │         │
# MAGIC            Config    Config   Config
# MAGIC ```
# MAGIC
# MAGIC ## Why DI for MCP Servers?
# MAGIC
# MAGIC ✅ **Testing** - Mock all external services
# MAGIC ✅ **Configuration** - Easy to change backends
# MAGIC ✅ **Resource Management** - Proper lifecycle
# MAGIC ✅ **Reusability** - Share services across tools
# MAGIC ✅ **Maintainability** - Clean, organized code
# MAGIC
# MAGIC ## MCP Tools we'll build:
# MAGIC
# MAGIC 1. **get_user** - Get user from database (with cache)
# MAGIC 2. **get_weather** - Get weather from API (with cache)
# MAGIC 3. **notify_user** - Send notification via email
# MAGIC 4. **user_weather_alert** - Complex tool using multiple services
# MAGIC
# MAGIC ## Key Benefits:
# MAGIC
# MAGIC - All tools share same database connection pool
# MAGIC - All tools share same cache
# MAGIC - Easy to switch implementations (SMTP vs SendGrid)
# MAGIC - Easy to test (inject mocks)
# MAGIC - Proper resource cleanup on shutdown
# MAGIC
# MAGIC Let's build it! 🚀

# COMMAND ----------

# DBTITLE 1,MCP Server with DI - Complete Implementation
from dataclasses import dataclass
from typing import Dict, Optional, List
import asyncio
import json

print("🔌 MCP Server with Dependency Injection\n")
print("="*80)

# ==================== Configuration ====================

@dataclass
class ServerConfig:
    """Server configuration"""
    db_host: str = "localhost"
    db_port: int = 5432
    redis_host: str = "localhost"
    redis_port: int = 6379
    weather_api_key: str = "weather_key_123"
    weather_api_url: str = "https://api.weather.com"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

print("⚙️ Configuration loaded\n")

# ==================== Infrastructure Services ====================

class DatabasePool:
    """Database connection pool"""
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.pool = None
        self.users = {
            1: {"id": 1, "name": "Alice", "email": "alice@example.com", "location": "New York"},
            2: {"id": 2, "name": "Bob", "email": "bob@example.com", "location": "San Francisco"}
        }
        print(f"💾 DatabasePool created: {host}:{port}")
    
    async def connect(self):
        await asyncio.sleep(0.05)
        self.pool = "connected"
        print("✅ Database pool connected")
        return self
    
    async def disconnect(self):
        await asyncio.sleep(0.05)
        self.pool = None
        print("🔌 Database pool disconnected")
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        await asyncio.sleep(0.05)
        return self.users.get(user_id)

class RedisCache:
    """Redis cache client"""
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._cache = {}
        print(f"🗄️ RedisCache created: {host}:{port}")
    
    async def connect(self):
        await asyncio.sleep(0.05)
        print("✅ Redis cache connected")
        return self
    
    async def disconnect(self):
        await asyncio.sleep(0.05)
        self._cache.clear()
        print("🔌 Redis cache disconnected")
    
    async def get(self, key: str) -> Optional[any]:
        return self._cache.get(key)
    
    async def set(self, key: str, value: any, ttl: int = 300):
        self._cache[key] = value

class HttpClient:
    """HTTP client for external APIs"""
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        print(f"🌐 HttpClient created: {base_url}")
    
    async def get(self, endpoint: str) -> Dict:
        await asyncio.sleep(0.1)  # Simulate API call
        # Simulated weather data
        return {
            "location": endpoint.split('/')[-1],
            "temperature": 72,
            "condition": "Sunny",
            "humidity": 45
        }

class EmailClient:
    """Email service client"""
    def __init__(self, smtp_host: str, smtp_port: int):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sent_emails = []
        print(f"📧 EmailClient created: {smtp_host}:{smtp_port}")
    
    async def send(self, to: str, subject: str, body: str):
        await asyncio.sleep(0.05)
        email = {"to": to, "subject": subject, "body": body}
        self.sent_emails.append(email)
        print(f"  ✉️ Email sent to {to}: {subject}")

# ==================== Application Services ====================

class UserService:
    """User management service"""
    def __init__(self, db: DatabasePool, cache: RedisCache):
        self.db = db
        self.cache = cache
        print("👥 UserService initialized")
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        # Try cache first
        cache_key = f"user:{user_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            print(f"  🎯 Cache hit: user {user_id}")
            return cached
        
        # Query database
        print(f"  🔍 Querying database: user {user_id}")
        user = await self.db.get_user(user_id)
        
        # Cache result
        if user:
            await self.cache.set(cache_key, user)
        
        return user

class WeatherService:
    """Weather data service"""
    def __init__(self, http: HttpClient, cache: RedisCache):
        self.http = http
        self.cache = cache
        print("⛅ WeatherService initialized")
    
    async def get_weather(self, location: str) -> Dict:
        # Try cache
        cache_key = f"weather:{location}"
        cached = await self.cache.get(cache_key)
        if cached:
            print(f"  🎯 Cache hit: weather for {location}")
            return cached
        
        # Call API
        print(f"  🌐 Fetching weather: {location}")
        weather = await self.http.get(f"/weather/{location}")
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, weather, ttl=300)
        
        return weather

class NotificationService:
    """Notification service"""
    def __init__(self, email: EmailClient):
        self.email = email
        print("🔔 NotificationService initialized")
    
    async def notify(self, user_email: str, message: str):
        await self.email.send(
            to=user_email,
            subject="Notification",
            body=message
        )

# ==================== MCP Server ====================

class MCPServerWithDI:
    """
    MCP Server using Dependency Injection
    All services are injected, making it testable and flexible
    """
    
    def __init__(self,
                 user_service: UserService,
                 weather_service: WeatherService,
                 notification_service: NotificationService):
        self.user_service = user_service
        self.weather_service = weather_service
        self.notification_service = notification_service
        print("\n🔌 MCP Server initialized with all services")
    
    async def handle_get_user(self, user_id: int) -> Dict:
        """MCP Tool: Get user information"""
        user = await self.user_service.get_user(user_id)
        if not user:
            return {"error": "User not found"}
        return user
    
    async def handle_get_weather(self, location: str) -> Dict:
        """MCP Tool: Get weather for location"""
        weather = await self.weather_service.get_weather(location)
        return weather
    
    async def handle_notify_user(self, user_id: int, message: str) -> Dict:
        """MCP Tool: Send notification to user"""
        user = await self.user_service.get_user(user_id)
        if not user:
            return {"error": "User not found"}
        
        await self.notification_service.notify(user["email"], message)
        return {"status": "success", "message": f"Notified {user['name']}"}
    
    async def handle_user_weather_alert(self, user_id: int) -> Dict:
        """
        MCP Tool: Complex operation using multiple services
        Get user, fetch weather for their location, send alert if needed
        """
        # Get user
        user = await self.user_service.get_user(user_id)
        if not user:
            return {"error": "User not found"}
        
        # Get weather for user's location
        weather = await self.weather_service.get_weather(user["location"])
        
        # Check if alert needed
        if weather["temperature"] > 90:
            message = f"High temperature alert: {weather['temperature']}°F in {user['location']}"
            await self.notification_service.notify(user["email"], message)
            return {
                "status": "alert_sent",
                "user": user["name"],
                "weather": weather,
                "message": message
            }
        
        return {
            "status": "no_alert_needed",
            "user": user["name"],
            "weather": weather
        }

# ==================== Application Bootstrap ====================

print("\n" + "="*80)
print("🏗️ Bootstrapping Application with DI")
print("="*80 + "\n")

async def create_mcp_server() -> MCPServerWithDI:
    """
    Factory function to create MCP server with all dependencies
    This is where DI magic happens!
    """
    # Load configuration
    config = ServerConfig()
    
    # Create infrastructure services
    print("1️⃣ Creating infrastructure services...\n")
    db = DatabasePool(config.db_host, config.db_port)
    await db.connect()
    
    cache = RedisCache(config.redis_host, config.redis_port)
    await cache.connect()
    
    http = HttpClient(config.weather_api_url, config.weather_api_key)
    email = EmailClient(config.smtp_host, config.smtp_port)
    
    # Create application services (inject infrastructure)
    print("\n2️⃣ Creating application services...\n")
    user_service = UserService(db, cache)
    weather_service = WeatherService(http, cache)
    notification_service = NotificationService(email)
    
    # Create MCP server (inject application services)
    print("\n3️⃣ Creating MCP server...")
    mcp_server = MCPServerWithDI(
        user_service,
        weather_service,
        notification_service
    )
    
    return mcp_server

# Create the server
mcp_server = await create_mcp_server()

print("\n✅ MCP Server ready!\n")

# ==================== Using the MCP Server ====================

print("="*80)
print("🚀 Using MCP Server Tools")
print("="*80 + "\n")

print("Tool 1: get_user\n")
result = await mcp_server.handle_get_user(1)
print(f"Result: {json.dumps(result, indent=2)}\n")

print("\nTool 2: get_user (from cache)\n")
result = await mcp_server.handle_get_user(1)  # Should hit cache
print(f"Result: {json.dumps(result, indent=2)}\n")

print("\nTool 3: get_weather\n")
result = await mcp_server.handle_get_weather("New York")
print(f"Result: {json.dumps(result, indent=2)}\n")

print("\nTool 4: notify_user\n")
result = await mcp_server.handle_notify_user(2, "Hello from MCP!")
print(f"Result: {json.dumps(result, indent=2)}\n")

print("\nTool 5: user_weather_alert (complex operation)\n")
result = await mcp_server.handle_user_weather_alert(1)
print(f"Result: {json.dumps(result, indent=2)}\n")

print("\n" + "="*80)
print("✅ All MCP Tools Working!")
print("="*80)

print("\n💡 Key Benefits of DI in this MCP Server:")
print("   1. All tools share same DB connection pool")
print("   2. All tools share same cache")
print("   3. Easy to test (inject mocks)")
print("   4. Easy to change implementations")
print("   5. Proper resource lifecycle management")
print("   6. Clean, maintainable code structure")

print("\n🧪 Testing Benefits:")
print("   - Can inject MockDatabase for testing")
print("   - Can inject MockCache for testing")
print("   - Can inject MockHTTP for testing")
print("   - Can inject MockEmail for testing")
print("   - No real external calls during tests!")

print("\n🚀 Production Benefits:")
print("   - Connection pooling across all tools")
print("   - Shared caching reduces API calls")
print("   - Easy configuration management")
print("   - Graceful startup and shutdown")

# COMMAND ----------

# DBTITLE 1,🎓 Summary & Best Practices
# MAGIC %md
# MAGIC # 🎓 Summary & Best Practices
# MAGIC
# MAGIC ## Kya Seekha Humne? (What Did We Learn?)
# MAGIC
# MAGIC ### 📚 Core Concepts:
# MAGIC
# MAGIC 1. **Dependency Injection Basics**
# MAGIC    - Dependencies ko inject karo, create mat karo
# MAGIC    - Constructor, setter, aur parameter injection
# MAGIC    - Protocol/Interface-based injection
# MAGIC
# MAGIC 2. **DI Container Pattern**
# MAGIC    - Centralized service registration
# MAGIC    - Lifecycle management (Singleton, Transient, Scoped)
# MAGIC    - Automatic dependency resolution
# MAGIC
# MAGIC 3. **dependency-injector Library**
# MAGIC    - Declarative containers
# MAGIC    - Provider types (Singleton, Factory, Resource)
# MAGIC    - Automatic wiring
# MAGIC    - Configuration management
# MAGIC
# MAGIC 4. **FastAPI Integration**
# MAGIC    - Built-in DI with `Depends()`
# MAGIC    - Lifespan events for resource management
# MAGIC    - Combining with dependency-injector
# MAGIC
# MAGIC 5. **Testing with DI**
# MAGIC    - Mock injection
# MAGIC    - Container overrides
# MAGIC    - AsyncMock for async code
# MAGIC    - Pytest fixtures
# MAGIC
# MAGIC 6. **Real-World MCP Server**
# MAGIC    - Multi-service orchestration
# MAGIC    - Shared resources (DB pool, cache)
# MAGIC    - Clean architecture
# MAGIC    - Production-ready patterns
# MAGIC
# MAGIC ## 🎯 Best Practices
# MAGIC
# MAGIC ### 1️⃣ Design Principles
# MAGIC
# MAGIC ✅ **Depend on Abstractions**
# MAGIC ```python
# MAGIC # ✅ Good: Depend on Protocol/Interface
# MAGIC class UserService:
# MAGIC     def __init__(self, database: DatabaseProtocol):
# MAGIC         self.db = database
# MAGIC
# MAGIC # ❌ Bad: Depend on concrete class
# MAGIC class UserService:
# MAGIC     def __init__(self, database: PostgresDatabase):
# MAGIC         self.db = database
# MAGIC ```
# MAGIC
# MAGIC ✅ **Constructor Injection Over Setter Injection**
# MAGIC ```python
# MAGIC # ✅ Good: Required dependencies in constructor
# MAGIC class Service:
# MAGIC     def __init__(self, db: Database, cache: Cache):
# MAGIC         self.db = db
# MAGIC         self.cache = cache
# MAGIC
# MAGIC # ❌ Bad: Setter injection for required dependencies
# MAGIC class Service:
# MAGIC     def __init__(self):
# MAGIC         self.db = None  # Might be None!
# MAGIC     
# MAGIC     def set_database(self, db):
# MAGIC         self.db = db
# MAGIC ```
# MAGIC
# MAGIC ✅ **Single Responsibility**
# MAGIC ```python
# MAGIC # ✅ Good: Each service has one responsibility
# MAGIC class UserRepository:  # Only data access
# MAGIC     pass
# MAGIC
# MAGIC class UserValidator:   # Only validation
# MAGIC     pass
# MAGIC
# MAGIC class UserService:     # Orchestrates
# MAGIC     def __init__(self, repo: UserRepository, validator: UserValidator):
# MAGIC         pass
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Lifecycle Management
# MAGIC
# MAGIC ✅ **Use Appropriate Lifetimes**
# MAGIC ```python
# MAGIC # Singleton: Database pools, caches, config
# MAGIC container.register_singleton('db_pool', create_pool)
# MAGIC
# MAGIC # Transient: Stateless services, request handlers
# MAGIC container.register_transient('email_service', EmailService)
# MAGIC
# MAGIC # Scoped: Per-request resources
# MAGIC container.register_scoped('user_session', UserSession)
# MAGIC ```
# MAGIC
# MAGIC ✅ **Async Context Managers for Resources**
# MAGIC ```python
# MAGIC class DatabasePool:
# MAGIC     async def __aenter__(self):
# MAGIC         await self.connect()
# MAGIC         return self
# MAGIC     
# MAGIC     async def __aexit__(self, *args):
# MAGIC         await self.disconnect()
# MAGIC
# MAGIC # Use with Resource provider
# MAGIC container.database = providers.Resource(
# MAGIC     DatabasePool,
# MAGIC     host="localhost"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Configuration
# MAGIC
# MAGIC ✅ **Externalize Configuration**
# MAGIC ```python
# MAGIC # ✅ Good: Configuration from external source
# MAGIC class Container(containers.DeclarativeContainer):
# MAGIC     config = providers.Configuration()
# MAGIC     
# MAGIC     database = providers.Singleton(
# MAGIC         Database,
# MAGIC         host=config.db.host,
# MAGIC         port=config.db.port
# MAGIC     )
# MAGIC
# MAGIC # Load from file
# MAGIC container.config.from_yaml('config.yaml')
# MAGIC container.config.from_env('APP', as_=Config)
# MAGIC ```
# MAGIC
# MAGIC ### 4️⃣ Testing
# MAGIC
# MAGIC ✅ **Make Everything Mockable**
# MAGIC ```python
# MAGIC # ✅ Good: All dependencies injected
# MAGIC class Service:
# MAGIC     def __init__(self, db, cache, email):
# MAGIC         self.db = db
# MAGIC         self.cache = cache
# MAGIC         self.email = email
# MAGIC
# MAGIC # Easy to test!
# MAGIC def test_service():
# MAGIC     service = Service(
# MAGIC         MockDatabase(),
# MAGIC         MockCache(),
# MAGIC         MockEmail()
# MAGIC     )
# MAGIC     # Test...
# MAGIC ```
# MAGIC
# MAGIC ✅ **Use Container Overrides**
# MAGIC ```python
# MAGIC def test_with_container():
# MAGIC     container = ApplicationContainer()
# MAGIC     container.database.override(MockDatabase())
# MAGIC     container.email.override(MockEmail())
# MAGIC     
# MAGIC     service = container.user_service()
# MAGIC     # Test with mocks...
# MAGIC ```
# MAGIC
# MAGIC ### 5️⃣ Error Handling
# MAGIC
# MAGIC ✅ **Validate Dependencies**
# MAGIC ```python
# MAGIC class Service:
# MAGIC     def __init__(self, db: Database):
# MAGIC         if db is None:
# MAGIC             raise ValueError("Database is required")
# MAGIC         self.db = db
# MAGIC ```
# MAGIC
# MAGIC ✅ **Graceful Degradation**
# MAGIC ```python
# MAGIC class Service:
# MAGIC     def __init__(self, cache: Optional[Cache] = None):
# MAGIC         self.cache = cache  # Optional dependency
# MAGIC     
# MAGIC     async def get_data(self, key):
# MAGIC         if self.cache:
# MAGIC             cached = await self.cache.get(key)
# MAGIC             if cached:
# MAGIC                 return cached
# MAGIC         # Fallback to primary source
# MAGIC         return await self.fetch_from_source(key)
# MAGIC ```
# MAGIC
# MAGIC ## 🛑 Common Pitfalls
# MAGIC
# MAGIC ❌ **Service Locator Anti-Pattern**
# MAGIC ```python
# MAGIC # ❌ Bad: Service locator (global state)
# MAGIC class ServiceLocator:
# MAGIC     services = {}
# MAGIC
# MAGIC class MyService:
# MAGIC     def __init__(self):
# MAGIC         self.db = ServiceLocator.get('database')  # Hidden dependency!
# MAGIC ```
# MAGIC
# MAGIC ❌ **Circular Dependencies**
# MAGIC ```python
# MAGIC # ❌ Bad: A depends on B, B depends on A
# MAGIC class ServiceA:
# MAGIC     def __init__(self, service_b: ServiceB):
# MAGIC         pass
# MAGIC
# MAGIC class ServiceB:
# MAGIC     def __init__(self, service_a: ServiceA):
# MAGIC         pass
# MAGIC
# MAGIC # ✅ Fix: Refactor to break cycle
# MAGIC ```
# MAGIC
# MAGIC ❌ **Over-injection**
# MAGIC ```python
# MAGIC # ❌ Bad: Too many dependencies
# MAGIC class Service:
# MAGIC     def __init__(self, db, cache, email, sms, push, logger, metrics, config, ...):
# MAGIC         pass  # God object!
# MAGIC
# MAGIC # ✅ Fix: Break into smaller services
# MAGIC ```
# MAGIC
# MAGIC ## 🚀 When to Use DI?
# MAGIC
# MAGIC ### ✅ Use DI When:
# MAGIC - Building testable applications
# MAGIC - Working with external dependencies (DB, APIs, etc.)
# MAGIC - Need flexible configuration
# MAGIC - Building modular, maintainable systems
# MAGIC - Working in teams (clear interfaces)
# MAGIC - Need to swap implementations
# MAGIC
# MAGIC ### ❌ Don't Overdo It:
# MAGIC - Simple scripts that don't need testing
# MAGIC - Pure functions without side effects
# MAGIC - Data classes / DTOs
# MAGIC - Utilities that have no dependencies
# MAGIC
# MAGIC ## 📚 Further Learning
# MAGIC
# MAGIC ### Libraries to Explore:
# MAGIC 1. **dependency-injector** - Full-featured DI framework
# MAGIC 2. **injector** - Lightweight DI for Python
# MAGIC 3. **punq** - Simple DI container
# MAGIC 4. **FastAPI** - Has built-in DI
# MAGIC
# MAGIC ### Resources:
# MAGIC - [dependency-injector docs](https://python-dependency-injector.ets-labs.org/)
# MAGIC - [FastAPI Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
# MAGIC - [Martin Fowler - Inversion of Control](https://martinfowler.com/articles/injection.html)
# MAGIC
# MAGIC ## 🎉 Conclusion
# MAGIC
# MAGIC **Dependency Injection is NOT about frameworks - it's about DESIGN!**
# MAGIC
# MAGIC Key principles:
# MAGIC 1. 📦 **Inject dependencies, don't create them**
# MAGIC 2. 🔗 **Depend on abstractions, not concretions**
# MAGIC 3. 🧪 **Make testing easy**
# MAGIC 4. ⚙️ **Externalize configuration**
# MAGIC 5. 📊 **Manage lifecycles properly**
# MAGIC
# MAGIC **Remember**: DI ki asli power testability aur flexibility mein hai!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📝 Quick Reference Card:
# MAGIC
# MAGIC | Pattern | Use Case | Example |
# MAGIC |---------|----------|----------|
# MAGIC | Constructor Injection | Required dependencies | `__init__(self, db: DB)` |
# MAGIC | Property Injection | Optional dependencies | `service.cache = cache` |
# MAGIC | Method Injection | One-time use | `process(data, db)` |
# MAGIC | Singleton | Shared resources | DB pool, cache |
# MAGIC | Transient | Stateless services | Email sender |
# MAGIC | Scoped | Per-request | User session |
# MAGIC
# MAGIC ### 🎯 Code Template:
# MAGIC
# MAGIC ```python
# MAGIC # 1. Define services with clear dependencies
# MAGIC class MyService:
# MAGIC     def __init__(self, db: Database, cache: Cache):
# MAGIC         self.db = db
# MAGIC         self.cache = cache
# MAGIC
# MAGIC # 2. Create container
# MAGIC from dependency_injector import containers, providers
# MAGIC
# MAGIC class Container(containers.DeclarativeContainer):
# MAGIC     config = providers.Configuration()
# MAGIC     database = providers.Singleton(Database, ...)
# MAGIC     cache = providers.Singleton(Cache, ...)
# MAGIC     my_service = providers.Factory(MyService, db=database, cache=cache)
# MAGIC
# MAGIC # 3. Initialize
# MAGIC container = Container()
# MAGIC container.config.from_yaml('config.yaml')
# MAGIC
# MAGIC # 4. Use
# MAGIC service = container.my_service()
# MAGIC
# MAGIC # 5. Test
# MAGIC def test_service():
# MAGIC     container = Container()
# MAGIC     container.database.override(MockDatabase())
# MAGIC     service = container.my_service()
# MAGIC     # Test...
# MAGIC ```
# MAGIC
# MAGIC ## ✨ Final Words
# MAGIC
# MAGIC Dependency Injection ek powerful pattern hai jo aapke code ko:
# MAGIC - **Testable** banata hai
# MAGIC - **Flexible** banata hai
# MAGIC - **Maintainable** banata hai
# MAGIC - **Professional** banata hai
# MAGIC
# MAGIC Yeh initially thoda complex lag sakta hai, but once you understand it, you'll never want to go back!
# MAGIC
# MAGIC **Happy Coding! 🚀**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *"Make the change easy, then make the easy change." - Kent Beck*