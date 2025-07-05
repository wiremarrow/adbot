"""
Base model classes for AdBot
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID


Base = declarative_base()


class BaseModel(Base):
    """Base model with common fields"""
    
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update(self, **kwargs) -> None:
        """Update model attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
    
    @classmethod
    def create(cls, session: Session, **kwargs) -> "BaseModel":
        """Create and save a new instance"""
        instance = cls(**kwargs)
        session.add(instance)
        session.commit()
        session.refresh(instance)
        return instance
    
    def save(self, session: Session) -> None:
        """Save the instance to database"""
        session.add(self)
        session.commit()
        session.refresh(self)
    
    def delete(self, session: Session) -> None:
        """Delete the instance from database"""
        session.delete(self)
        session.commit()


class TimestampMixin:
    """Mixin for timestamp fields"""
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class MetadataMixin:
    """Mixin for metadata fields"""
    
    metadata = Column(JSON, default=dict, nullable=False)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)