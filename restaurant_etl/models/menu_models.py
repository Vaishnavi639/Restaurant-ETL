from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re


class MenuItem(BaseModel):
    item_name: str = Field(
        ...,  # "..." = required field
        description="The name of the dish (e.g., 'Paneer Tikka', 'Margherita Pizza')",
        min_length=2,
        examples=["Butter Chicken", "Caesar Salad"]
    )
    
    category: Optional[str] = Field(
        None,
        description="Category like 'Appetizers', 'Main Course', 'Desserts', 'Beverages'",
        examples=["Appetizers", "Main Course", "Vegetarian Specialties"]
    )
    
    description: Optional[str] = Field(
        None,
        description="Detailed description of the dish including ingredients and cooking method",
        examples=["Tender chicken pieces cooked in rich butter gravy with cream"]
    )
    price: Optional[float] = Field(
        None,
        ge=0,  
        description="Single price for the item in numeric format (no currency symbols)"
    )
    
    half_plate_price: Optional[float] = Field(
        None,
        ge=0,
        description="Price for half portion"
    )
    full_plate_price: Optional[float] = Field(
        None,
        ge=0,
        description="Price for full portion"
    )
    
    small_price: Optional[float] = Field(None, ge=0, description="Price for small size")
    medium_price: Optional[float] = Field(None, ge=0, description="Price for medium size")
    large_price: Optional[float] = Field(None, ge=0, description="Price for large size")

    currency: Optional[str] = Field(
        None,
        description="Currency symbol found in the menu ($, ₹, €, £, etc.)",
        examples=["$", "₹", "€", "£"]
    )
    
    spice_level: Optional[str] = Field(
        None,
        description="Spice level if mentioned (Mild, Medium, Hot, Very Hot)",
        examples=["Mild", "Medium", "Hot"]
    )
    
    dietary_tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Dietary information (Vegan, Vegetarian, Gluten-Free, etc.)",
        examples=[["Vegan"], ["Vegetarian", "Gluten-Free"]]
    )

    @field_validator('item_name')
    @classmethod
    def clean_name(cls, v: str) -> str:
        v = ' '.join(v.split())
        return v.strip()
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        valid_currencies = ['$', '₹', '€', '£', '¥', 'AUD', 'USD', 'INR', 'EUR', 'GBP']
        if v in valid_currencies or len(v) <= 3:
            return v
        
        return None
    
    @field_validator('price', 'half_plate_price', 'full_plate_price',
                     'small_price', 'medium_price', 'large_price')
    @classmethod
    def validate_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if v < 0:
                raise ValueError("Price cannot be negative")
            if v > 100000:  # Increased for different currencies
                raise ValueError(f"Price {v} seems unreasonably high")
        return v

    def has_any_price(self) -> bool:
        """Check if item has at least one price."""
        return any([
            self.price is not None,
            self.half_plate_price is not None,
            self.full_plate_price is not None,
            self.small_price is not None,
            self.medium_price is not None,
            self.large_price is not None
        ])
    
    def get_price_display(self) -> str:

        currency_symbol = self.currency or "$"  

        if self.price is not None:
            return f"{currency_symbol}{self.price:.2f}"

        parts = []
        if self.half_plate_price is not None:
            parts.append(f"Half: {currency_symbol}{self.half_plate_price:.2f}")
        if self.full_plate_price is not None:
            parts.append(f"Full: {currency_symbol}{self.full_plate_price:.2f}")

        if self.small_price is not None:
            parts.append(f"Small: {currency_symbol}{self.small_price:.2f}")
        if self.medium_price is not None:
            parts.append(f"Medium: {currency_symbol}{self.medium_price:.2f}")
        if self.large_price is not None:
            parts.append(f"Large: {currency_symbol}{self.large_price:.2f}")
        
        return " | ".join(parts) if parts else "No price"
    
    def get_primary_price(self) -> Optional[float]:
        return (
            self.price or
            self.full_plate_price or
            self.half_plate_price or
            self.large_price or
            self.medium_price or
            self.small_price
        )


class MenuData(BaseModel):


    items: List[MenuItem] = Field(
        default_factory=list,
        description="List of all menu items extracted"
    )
    
    restaurant_name: Optional[str] = Field(
        None,
        description="Name of the restaurant"
    )
    
    # Metadata fields
    total_items: int = Field(
        0,
        description="Total number of items extracted"
    )
    
    source_file: Optional[str] = Field(
        None,
        description="Source file name"
    )
    
    extraction_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the extraction"
    )
    
    detected_currency: Optional[str] = Field(
        None,
        description="Primary currency detected in the menu"
    )
    
    extraction_method: Optional[str] = Field(
        None,
        description="Method used for extraction (text, vision, ocr)"
    )
    
    def to_dataframe(self):
        import pandas as pd
        
        if not self.items:
            return pd.DataFrame()
        data = []
        for item in self.items:
            row = {
                'item_name': item.item_name,
                'category': item.category,
                'description': item.description,
                'price': item.price,
                'half_plate_price': item.half_plate_price,
                'full_plate_price': item.full_plate_price,
                'small_price': item.small_price,
                'medium_price': item.medium_price,
                'large_price': item.large_price,
                'currency': item.currency or self.detected_currency,
                'price_display': item.get_price_display(),
                'spice_level': item.spice_level,
                'dietary_tags': ', '.join(item.dietary_tags) if item.dietary_tags else None
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        column_order = [
            'item_name',
            'category', 
            'description',
            'price_display',
            'price',
            'half_plate_price',
            'full_plate_price',
            'small_price',
            'medium_price',
            'large_price',
            'currency',
            'spice_level',
            'dietary_tags'
        ]

        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        return df
    
    def get_summary(self) -> dict:

        categories = {}
        total_with_prices = 0
        price_range = []
        
        for item in self.items:
            cat = item.category or "Uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
            
            # Track prices
            if item.has_any_price():
                total_with_prices += 1
                primary_price = item.get_primary_price()
                if primary_price:
                    price_range.append(primary_price)
        
        return {
            'total_items': len(self.items),
            'items_with_prices': total_with_prices,
            'categories': categories,
            'price_range': {
                'min': min(price_range) if price_range else None,
                'max': max(price_range) if price_range else None,
                'avg': sum(price_range) / len(price_range) if price_range else None
            },
            'currency': self.detected_currency
        }


