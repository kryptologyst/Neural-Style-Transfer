"""
Database models for storing style transfer results and metadata.
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import json
import base64
from typing import Optional, Dict, Any

Base = declarative_base()


class StyleTransferResult(Base):
    """Model for storing style transfer results"""
    __tablename__ = 'style_transfer_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content_image_path = Column(String(500), nullable=False)
    style_image_path = Column(String(500), nullable=False)
    output_image_path = Column(String(500), nullable=False)
    method = Column(String(50), nullable=False)  # 'adain', 'optimization', etc.
    parameters = Column(Text)  # JSON string of parameters used
    processing_time = Column(Float)  # Time taken in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'content_image_path': self.content_image_path,
            'style_image_path': self.style_image_path,
            'output_image_path': self.output_image_path,
            'method': self.method,
            'parameters': json.loads(self.parameters) if self.parameters else {},
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class StyleTemplate(Base):
    """Model for storing popular style templates"""
    __tablename__ = 'style_templates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    style_image_path = Column(String(500), nullable=False)
    thumbnail_path = Column(String(500))
    category = Column(String(50))  # 'classical', 'modern', 'abstract', etc.
    popularity_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'style_image_path': self.style_image_path,
            'thumbnail_path': self.thumbnail_path,
            'category': self.category,
            'popularity_score': self.popularity_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class UserSession(Base):
    """Model for tracking user sessions and preferences"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, unique=True)
    preferences = Column(Text)  # JSON string of user preferences
    total_transfers = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'preferences': json.loads(self.preferences) if self.preferences else {},
            'total_transfers': self.total_transfers,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }


class DatabaseManager:
    """Database manager for style transfer application"""
    
    def __init__(self, database_url: str = "sqlite:///style_transfer.db"):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_transfer_result(self, content_path: str, style_path: str, 
                           output_path: str, method: str, parameters: Dict[str, Any],
                           processing_time: float) -> int:
        """Save style transfer result to database"""
        session = self.get_session()
        try:
            result = StyleTransferResult(
                content_image_path=content_path,
                style_image_path=style_path,
                output_image_path=output_path,
                method=method,
                parameters=json.dumps(parameters),
                processing_time=processing_time
            )
            session.add(result)
            session.commit()
            return result.id
        finally:
            session.close()
    
    def get_transfer_results(self, limit: int = 50, offset: int = 0):
        """Get style transfer results with pagination"""
        session = self.get_session()
        try:
            results = session.query(StyleTransferResult)\
                           .order_by(StyleTransferResult.created_at.desc())\
                           .limit(limit)\
                           .offset(offset)\
                           .all()
            return [result.to_dict() for result in results]
        finally:
            session.close()
    
    def get_transfer_result_by_id(self, result_id: int):
        """Get specific transfer result by ID"""
        session = self.get_session()
        try:
            result = session.query(StyleTransferResult).filter_by(id=result_id).first()
            return result.to_dict() if result else None
        finally:
            session.close()
    
    def add_style_template(self, name: str, description: str, style_image_path: str,
                          thumbnail_path: str = None, category: str = None) -> int:
        """Add a new style template"""
        session = self.get_session()
        try:
            template = StyleTemplate(
                name=name,
                description=description,
                style_image_path=style_image_path,
                thumbnail_path=thumbnail_path,
                category=category
            )
            session.add(template)
            session.commit()
            return template.id
        finally:
            session.close()
    
    def get_style_templates(self, category: str = None):
        """Get style templates, optionally filtered by category"""
        session = self.get_session()
        try:
            query = session.query(StyleTemplate)
            if category:
                query = query.filter_by(category=category)
            templates = query.order_by(StyleTemplate.popularity_score.desc()).all()
            return [template.to_dict() for template in templates]
        finally:
            session.close()
    
    def update_template_popularity(self, template_id: int, increment: float = 1.0):
        """Update template popularity score"""
        session = self.get_session()
        try:
            template = session.query(StyleTemplate).filter_by(id=template_id).first()
            if template:
                template.popularity_score += increment
                session.commit()
        finally:
            session.close()
    
    def create_user_session(self, session_id: str, preferences: Dict[str, Any] = None) -> int:
        """Create a new user session"""
        session = self.get_session()
        try:
            user_session = UserSession(
                session_id=session_id,
                preferences=json.dumps(preferences or {})
            )
            session.add(user_session)
            session.commit()
            return user_session.id
        finally:
            session.close()
    
    def update_user_session(self, session_id: str, preferences: Dict[str, Any] = None,
                          increment_transfers: bool = False):
        """Update user session"""
        session = self.get_session()
        try:
            user_session = session.query(UserSession).filter_by(session_id=session_id).first()
            if user_session:
                if preferences:
                    user_session.preferences = json.dumps(preferences)
                if increment_transfers:
                    user_session.total_transfers += 1
                user_session.last_active = datetime.utcnow()
                session.commit()
        finally:
            session.close()


def create_mock_data(db_manager: DatabaseManager):
    """Create mock data for testing"""
    
    # Add style templates
    templates = [
        {
            'name': 'Van Gogh - Starry Night',
            'description': 'Classic swirling brushstrokes and vibrant colors',
            'style_image_path': 'assets/styles/starry_night.jpg',
            'category': 'classical'
        },
        {
            'name': 'Picasso - Cubist',
            'description': 'Geometric shapes and abstract forms',
            'style_image_path': 'assets/styles/picasso_cubist.jpg',
            'category': 'modern'
        },
        {
            'name': 'Monet - Water Lilies',
            'description': 'Impressionist style with soft, flowing colors',
            'style_image_path': 'assets/styles/monet_lilies.jpg',
            'category': 'impressionist'
        },
        {
            'name': 'Kandinsky - Abstract',
            'description': 'Bold colors and abstract geometric patterns',
            'style_image_path': 'assets/styles/kandinsky_abstract.jpg',
            'category': 'abstract'
        },
        {
            'name': 'Japanese Ukiyo-e',
            'description': 'Traditional Japanese woodblock print style',
            'style_image_path': 'assets/styles/ukiyo_e.jpg',
            'category': 'traditional'
        }
    ]
    
    for template in templates:
        db_manager.add_style_template(**template)
    
    print("Mock data created successfully!")


if __name__ == "__main__":
    # Initialize database and create mock data
    db_manager = DatabaseManager()
    create_mock_data(db_manager)
