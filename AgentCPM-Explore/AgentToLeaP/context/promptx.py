import re

class PromptX():

    template: str
    replacements: dict

    def __init__(self, prompt: str):
        """
        Initialize PromptTemplate instance.
        :param template: Original template text with <key>content</key> tags
        :param replacements: Dictionary for replacing tag content, e.g. {"tool_info": "xxx"}. If not provided, will be automatically parsed from template.
        """

        # Automatically parse all <key>content</key> in template and fill replacements dict
        self.replacements = self._parse_tags(prompt)
        # Clear tag content in template
        self.template = self._clear_tag_content(prompt)

    def _parse_tags(self, text: str) -> dict:
        """
        Parse all <key>content</key> tags in text, return {key: content} dictionary.
        :param text: Text to parse
        :return: Parsed dictionary
        """
        pattern = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
        return {match.group(1): match.group(2) for match in pattern.finditer(text)}

    def _clear_tag_content(self, text: str) -> str:
        """
        Replace all <key>content</key> in text with <key></key>, keeping only tag structure.
        :param text: Original text
        :return: Text with tag content cleared
        """
        pattern = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
        return pattern.sub(lambda m: f"<{m.group(1)}></{m.group(1)}>", text)

    def __getitem__(self, key):
        """
        Allow getting values from current replacement dict via obj[key].
        :param key: Tag name
        :return: Value corresponding to key in replacement dict, returns empty string if not exist
        """
        return self.replacements.get(key, "")

    def __setitem__(self, key, value):
        """
        Allow setting or updating values in replacement dict via obj[key] = value.
        :param key: Tag name
        :param value: New content to replace
        """
        self.replacements[key] = value

    def __str__(self):
        """
        When converting object to string (like str(obj) or f"{obj}"),
        automatically replace <key></key> tag content in template with values from dict.
        If there are keys in replacements that don't appear in template, append at the end.
        :return: Complete string after replacement
        """
        def replacer(match):
            tag = match.group(1)
            content = self.replacements.get(tag, "")
       
            if not content.startswith('\n'):
                content = '\n' + content
            if not content.endswith('\n'):
                content = content + '\n'
            return f"<{tag}>{content}</{tag}>"
        pattern = re.compile(r"<(\w+)></\1>", re.DOTALL)
        result = pattern.sub(replacer, self.template)

        template_keys = set(match.group(1) for match in pattern.finditer(self.template))
        extra_keys = [k for k in self.replacements if k not in template_keys]
        if extra_keys:
            extra_content = ""
            for k in extra_keys:
                content = self.replacements[k]
                if not content.startswith('\n'):
                    content = '\n' + content
                if not content.endswith('\n'):
                    content = content + '\n'
                extra_content += f"<{k}>{content}</{k}>"
            result += extra_content
        return result

    def update(self, new_replacements: dict):
        """
        Batch update content in replacement dict.
        :param new_replacements: New replacement content dict
        """
        # Force convert all values to string
        self.replacements.update({k: str(v) for k, v in new_replacements.items()})
