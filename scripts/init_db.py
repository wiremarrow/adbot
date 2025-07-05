#!/usr/bin/env python3
"""
Database initialization script for AdBot
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.models.base import Base
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger, get_logger


async def create_database_schema():
    """Create database schema"""
    
    # Setup logging
    setup_logger("adbot.db_init", "INFO", "logs/db_init.log")
    logger = get_logger("adbot.db_init")
    
    try:
        # Get configuration
        config_manager = ConfigManager()
        app_config = config_manager.get_app_config()
        
        logger.info("Initializing database schema")
        logger.info(f"Database URL: {app_config.database.host}:{app_config.database.port}/{app_config.database.name}")
        
        # Create async engine
        engine = create_async_engine(
            app_config.database.async_url,
            echo=True,  # Log SQL statements
            pool_size=app_config.database.pool_size,
            max_overflow=0
        )
        
        # Create all tables
        async with engine.begin() as conn:
            logger.info("Creating database tables...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        
        # Verify tables were created
        async with engine.begin() as conn:
            # Get table names
            result = await conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"Created {len(tables)} tables:")
            for table in sorted(tables):
                logger.info(f"  - {table}")
        
        await engine.dispose()
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def create_sample_data():
    """Create sample data for development"""
    
    logger = get_logger("adbot.db_init")
    
    try:
        config_manager = ConfigManager()
        app_config = config_manager.get_app_config()
        
        engine = create_async_engine(app_config.database.async_url)
        
        # Create async session
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            logger.info("Creating sample data...")
            
            # Import models here to avoid circular imports
            from src.models.user import User, Account, Platform
            from src.models.campaign import Campaign, CampaignStatus
            
            # Create sample user
            user = User(
                email="admin@adbot.ai",
                username="admin",
                first_name="AdBot",
                last_name="Admin",
                password_hash="dummy_hash",  # In production, use proper hashing
                is_verified=True,
                role="admin"
            )
            session.add(user)
            await session.flush()  # Get user ID
            
            # Create sample account
            account = Account(
                name="Sample Account",
                description="Sample account for testing",
                user_id=user.id,
                status="active"
            )
            session.add(account)
            await session.flush()  # Get account ID
            
            # Create sample platform
            platform = Platform(
                name="google_ads",
                display_name="Google Ads",
                account_id=account.id,
                status="connected",
                platform_account_id="123-456-7890",
                platform_account_name="Sample Google Ads Account"
            )
            session.add(platform)
            
            # Create sample campaign
            campaign = Campaign(
                name="Sample Campaign",
                platform="google_ads",
                platform_id="12345",
                account_id=account.id,
                status=CampaignStatus.ACTIVE,
                budget_type="daily",
                budget_amount=100.00,
                optimization_goal="conversions",
                bid_strategy="target_cpa"
            )
            session.add(campaign)
            
            await session.commit()
            
            logger.info("Sample data created:")
            logger.info(f"  - User: {user.email}")
            logger.info(f"  - Account: {account.name}")
            logger.info(f"  - Platform: {platform.display_name}")
            logger.info(f"  - Campaign: {campaign.name}")
        
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        raise


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize AdBot database")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create sample data for development"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating new ones"
    )
    
    args = parser.parse_args()
    
    if args.drop_existing:
        print("WARNING: This will drop all existing tables!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
        
        # Drop existing tables
        config_manager = ConfigManager()
        app_config = config_manager.get_app_config()
        engine = create_async_engine(app_config.database.async_url)
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            print("Existing tables dropped.")
        
        await engine.dispose()
    
    # Create schema
    await create_database_schema()
    
    # Create sample data if requested
    if args.sample_data:
        await create_sample_data()
    
    print("Database initialization completed!")


if __name__ == "__main__":
    asyncio.run(main())