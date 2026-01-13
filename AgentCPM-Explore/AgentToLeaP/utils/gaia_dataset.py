#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("gaia_dataset")

class GaiaDataset:
    """GAIA dataset loading and processing class"""
    
    def __init__(self, dataset_dir: Optional[str] = None):
        """
        Initialize GAIA dataset
        
        Args:
            dataset_dir: GAIA dataset directory, if None, use default path
        """
        if dataset_dir is None:
            # Use default path
            file_path = Path(__file__).resolve()
            script_dir = file_path.parent
            src_dir = script_dir.parent
            project_root = src_dir.parent
            
            # Default path is src/gaia_test/GAIA
            self.dataset_dir = script_dir / "GAIA"
        else:
            self.dataset_dir = Path(dataset_dir)
        
        # Verify if the dataset directory exists
        if not self.dataset_dir.exists():
            logger.warning(f"GAIA dataset directory does not exist: {self.dataset_dir}")
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created GAIA dataset directory: {self.dataset_dir}")
        
        # Initialize dataset
        self.datasets = {}
        
    def load_dataset(self, year: str = "2023", split: str = "validation") -> List[Dict[str, Any]]:
        """
        Load GAIA dataset
        
        Args:
            year: Dataset year
            split: Dataset split, e.g., validation or test
            
        Returns:
            List[Dict[str, Any]]: List of dataset samples
        """
        # Construct dataset path
        dataset_path = self.dataset_dir / year / split
        
        if not dataset_path.exists():
            logger.warning(f"GAIA dataset path does not exist: {dataset_path}")
            return []
        
        # Find all JSON files
        json_files = list(dataset_path.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {dataset_path}")
            return []
        
        # Load all JSON files
        dataset = []
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # If it's a list, extend the dataset
                    if isinstance(data, list):
                        dataset.extend(data)
                    # If it's a dictionary, append to the dataset
                    elif isinstance(data, dict):
                        dataset.append(data)
                    else:
                        logger.warning(f"Unrecognized data format: {json_file}")
                        
                logger.info(f"Loaded dataset file: {json_file}")
            except Exception as e:
                logger.error(f"Error loading dataset file {json_file}: {str(e)}")
        
        # Cache dataset
        cache_key = f"{year}_{split}"
        self.datasets[cache_key] = dataset
        
        logger.info(f"Loaded {len(dataset)} samples, year: {year}, split: {split}")
        
        return dataset
    
    def get_dataset(self, year: str = "2023", split: str = "validation") -> List[Dict[str, Any]]:
        """
        Get GAIA dataset, if loaded get from cache, otherwise load
        
        Args:
            year: Dataset year
            split: Dataset split, e.g., validation or test
            
        Returns:
            List[Dict[str, Any]]: List of dataset samples
        """
        cache_key = f"{year}_{split}"
        
        if cache_key in self.datasets:
            return self.datasets[cache_key]
        else:
            return self.load_dataset(year, split)
    
    def get_sample(self, index: int, year: str = "2023", split: str = "validation") -> Optional[Dict[str, Any]]:
        """
        Get sample by index
        
        Args:
            index: Sample index
            year: Dataset year
            split: Dataset split, e.g., validation or test
            
        Returns:
            Optional[Dict[str, Any]]: Sample data, returns None if not exists
        """
        dataset = self.get_dataset(year, split)
        
        if not dataset or index >= len(dataset):
            logger.warning(f"Sample index {index} out of range, dataset size: {len(dataset) if dataset else 0}")
            return None
        
        return dataset[index]
    
    def get_random_samples(self, count: int, year: str = "2023", split: str = "validation") -> List[Dict[str, Any]]:
        """
        Get random samples
        
        Args:
            count: Number of samples
            year: Dataset year
            split: Dataset split, e.g., validation or test
            
        Returns:
            List[Dict[str, Any]]: List of random samples
        """
        import random
        
        dataset = self.get_dataset(year, split)
        
        if not dataset:
            logger.warning(f"Dataset is empty, cannot get random samples")
            return []
        
        # Limit sample count
        count = min(count, len(dataset))
        
        # Randomly select samples
        return random.sample(dataset, count)
    
    def save_sample(self, sample: Dict[str, Any], year: str = "2023", split: str = "validation", filename: Optional[str] = None) -> bool:
        """
        Save sample to dataset
        
        Args:
            sample: Sample data
            year: Dataset year
            split: Dataset split, e.g., validation or test
            filename: Filename, if None, generate automatically
            
        Returns:
            bool: Whether save was successful
        """
        # Construct dataset path
        dataset_path = self.dataset_dir / year / split
        
        # Ensure directory exists
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if filename is None:
            import uuid
            filename = f"sample_{uuid.uuid4().hex[:8]}.json"
        
        # Ensure filename ends with .json
        if not filename.endswith(".json"):
            filename += ".json"
        
        # Construct file path
        file_path = dataset_path / filename
        
        try:
            # Save sample
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved sample to: {file_path}")
            
            # Update cache
            cache_key = f"{year}_{split}"
            if cache_key in self.datasets:
                self.datasets[cache_key].append(sample)
            
            return True
        except Exception as e:
            logger.error(f"Error saving sample: {str(e)}")
            return False
    
    def save_samples(self, samples: List[Dict[str, Any]], year: str = "2023", split: str = "validation", filename: Optional[str] = None) -> bool:
        """
        Save multiple samples to dataset
        
        Args:
            samples: List of sample data
            year: Dataset year
            split: Dataset split, e.g., validation or test
            filename: Filename, if None, generate automatically
            
        Returns:
            bool: Whether save was successful
        """
        # Construct dataset path
        dataset_path = self.dataset_dir / year / split
        
        # Ensure directory exists
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if filename is None:
            import uuid
            filename = f"samples_{uuid.uuid4().hex[:8]}.json"
        
        # Ensure filename ends with .json
        if not filename.endswith(".json"):
            filename += ".json"
        
        # Construct file path
        file_path = dataset_path / filename
        
        try:
            # Save samples
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(samples)} samples to: {file_path}")
            
            # Update cache
            cache_key = f"{year}_{split}"
            if cache_key in self.datasets:
                self.datasets[cache_key].extend(samples)
            
            return True
        except Exception as e:
            logger.error(f"Error saving samples: {str(e)}")
            return False 