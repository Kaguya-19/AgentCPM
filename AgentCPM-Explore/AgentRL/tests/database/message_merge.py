def is_same_context_message(message1, message2):
    """Check whether two messages have the same context.
    
    Possible keys in a message: role, content, audio, function_call, tool_calls, reasoning_content
    """
    for k in ["role", "content", "function_call", "tool_calls", "audio", "reasoning_content"]:
        if k not in message1 and k not in message2:
            continue

        if message1.get(k,None) is None:
            if message2.get(k,None) is not None:
                return False
            else:
                continue

        match k:
            case "content":
                if isinstance(message1[k], str) and isinstance(message2[k], str):
                    if message1[k] != message2[k]:
                        return False
                    
                elif isinstance(message1[k], list) and isinstance(message2[k], list):
                    if len(message1[k]) != len(message2[k]):
                        return False
                    
                    for item1, item2 in zip(message1[k], message2[k]):
                        if isinstance(item1, str) and isinstance(item2, str):
                            if item1 != item2:
                                return False
                        elif isinstance(item1, dict) and isinstance(item2, dict):
                            if item1.get("type") != item2.get("type"):
                                return False
                            if item1.get("type") == "text":
                                if item1.get("text") != item2.get("text"):
                                    return False
                            elif item1.get("type") == "image_rul":
                                if item1.get("image_url", {}).get("url") != item2.get("image_url", {}).get("url"):
                                    return False
                            else:
                                return False
                        else:
                            return False
                else:
                    if message1.get(k,"") != message2.get(k,""):
                        return False

            case "function_call":
                function_call1 = message1.get(k, {})
                function_call2 = message2.get(k, {})
                if function_call1.get("name") != function_call2.get("name"):
                    return False
                args1 = function_call1.get("arguments", {})
                args2 = function_call2.get("arguments", {})
                if args1 != args2:
                    return False

            case "tool_calls":
                tool_calls1 = message1.get(k, [])
                tool_calls2 = message2.get(k, [])
                if len(tool_calls1) != len(tool_calls2):
                    return False
                for tc1, tc2 in zip(tool_calls1, tool_calls2):
                    if tc1.get("type") != tc2.get("type"):
                        return False
                    if tc1.get("function") != tc2.get("function"):
                        return False

            case "role" | "reasoning_content" | "audio":
                if message1.get(k,None) != message2.get(k,None):
                    return False
                
            case _:
                raise ValueError(f"Unknown key in message: {k}")
    return True

def is_contained_in_prefix(s1,s2):
    if "tools" in s1["request"]:
        if "tools" not in s2["request"]:
            return False
        if s1["request"]["tools"] != s2["request"]["tools"]:
            return False
    
    if len(s1["request"]["messages"]) > len(s2["request"]["messages"]):
        return False

    for m1, m2 in zip(s1["request"]["messages"], s2["request"]["messages"]):
        if not is_same_context_message(m1, m2):
            return False
    
    if s1["response"]["choices"][0]["message"]:
        msg1 = s1["response"]["choices"][0]["message"]
        msg2 = s2["request"]["messages"][len(s1["request"]["messages"])]
        if not is_same_context_message(msg1, msg2):
            return False
    return True

# import pdb; pdb.set_trace()
import json
data = json.load(open("/nfsdata/luyaxi/AgentRL/tests/database/1.json"))
s1 = data[0]
s2 = data[1]
res = is_contained_in_prefix(s1,s2)
print(res)