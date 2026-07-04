# RAG评估模块 —— 基于Qwen的轻量级评估
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_client import get_qwen_llm
import json
import re


class RAGEvaluator:
    """基于LLM的RAG评估器，针对简历优化场景定制"""

    def __init__(self):
        self.llm = get_qwen_llm(temperature=0.1, max_tokens=1024)

    def evaluate_pipeline(
        self,
        original_resume: str,
        optimized_resume: str,
        jd_text: str,
        match_result: str,
        retrieved_chunks_count: int
    ) -> dict:
        """评估整个RAG流水线的质量"""

        # 截断过长文本避免token超限
        original = original_resume[:2000] if len(original_resume) > 2000 else original_resume
        optimized = optimized_resume[:2000] if len(optimized_resume) > 2000 else optimized_resume
        jd = jd_text[:1500] if len(jd_text) > 1500 else jd_text

        scores = {}

        # 1. 忠实度评估：优化后简历是否忠实于原始简历
        scores["faithfulness"] = self._evaluate_faithfulness(original, optimized)

        # 2. JD相关性评估：优化后简历与JD的匹配程度
        scores["jd_relevancy"] = self._evaluate_jd_relevancy(optimized, jd)

        # 3. 完整性评估：是否覆盖JD中的关键要求
        scores["completeness"] = self._evaluate_completeness(optimized, jd)

        # 4. 改进幅度评估：优化后相比原始简历的提升
        scores["improvement"] = self._evaluate_improvement(original, optimized, jd)

        # 5. 检索质量评估
        scores["retrieval_quality"] = self._evaluate_retrieval_quality(
            retrieved_chunks_count, jd
        )

        # 计算综合评分（加权平均）
        weights = {
            "faithfulness": 0.30,
            "jd_relevancy": 0.25,
            "completeness": 0.20,
            "improvement": 0.15,
            "retrieval_quality": 0.10
        }
        total = sum(scores[k] * weights[k] for k in weights)
        scores["overall"] = round(total, 1)

        # 生成评估总结
        scores["summary"] = self._generate_summary(scores)

        return scores

    def _evaluate_faithfulness(self, original: str, optimized: str) -> float:
        """评估忠实度：优化后的简历是否基于原始简历，没有编造内容"""
        prompt = f"""请评估优化后简历对原始简历的忠实度。

只输出一个0到10之间的整数数字，不要输出任何其他内容。

评分标准：
10分 = 完全忠实，所有内容都来自原始简历
7-9分 = 基本忠实，有少量合理的措辞优化
4-6分 = 部分忠实，存在一些原始简历中没有的信息
1-3分 = 大量编造，添加了很多原始简历中没有的经历或技能

原始简历（摘要）：{original}
优化后简历（摘要）：{optimized}"""

        return self._parse_score(self.llm.invoke(prompt).content)

    def _evaluate_jd_relevancy(self, optimized: str, jd: str) -> float:
        """评估JD相关性：优化后简历与JD的匹配程度"""
        prompt = f"""请评估优化后简历与岗位JD的相关性。

只输出一个0到10之间的整数数字，不要输出任何其他内容。

评分标准：
10分 = 高度匹配，简历内容完全覆盖JD核心要求
7-9分 = 较好匹配，大部分JD要求都有体现
4-6分 = 部分匹配，仅覆盖JD的部分要求
1-3分 = 匹配度低，简历内容与JD关联不大

优化后简历（摘要）：{optimized}
岗位JD：{jd}"""

        return self._parse_score(self.llm.invoke(prompt).content)

    def _evaluate_completeness(self, optimized: str, jd: str) -> float:
        """评估完整性：是否覆盖JD中的关键技能和要求"""
        prompt = f"""请评估优化后简历对JD要求的覆盖完整性。

只输出一个0到10之间的整数数字，不要输出任何其他内容。

评分标准：
10分 = 完全覆盖，JD中的所有关键技能和要求都有体现
7-9分 = 大部分覆盖，遗漏了少量非核心要求
4-6分 = 部分覆盖，缺少一些重要技能或要求
1-3分 = 覆盖不足，大部分JD要求未涉及

优化后简历（摘要）：{optimized}
岗位JD：{jd}"""

        return self._parse_score(self.llm.invoke(prompt).content)

    def _evaluate_improvement(self, original: str, optimized: str, jd: str) -> float:
        """评估改进幅度：优化后相比原始简历的提升程度"""
        prompt = f"""请评估优化后简历相比原始简历的改进幅度。

只输出一个0到10之间的整数数字，不要输出任何其他内容。

评分标准：
10分 = 改进显著，关键词匹配、表述质量、结构都有明显提升
7-9分 = 有明显改进，关键词和表述都有优化
4-6分 = 有一定改进，但提升不大
1-3分 = 改进很小，与原始简历差别不大

原始简历（摘要）：{original}
优化后简历（摘要）：{optimized}
岗位JD：{jd}"""

        return self._parse_score(self.llm.invoke(prompt).content)

    def _evaluate_retrieval_quality(self, retrieved_chunks_count: int, jd: str) -> float:
        """评估检索质量"""
        if retrieved_chunks_count >= 5:
            return 8.0
        elif retrieved_chunks_count >= 3:
            return 6.0
        elif retrieved_chunks_count >= 1:
            return 4.0
        else:
            return 2.0

    def _generate_summary(self, scores: dict) -> str:
        """生成评估总结文字"""
        overall = scores["overall"]
        if overall >= 8:
            level = "优秀"
        elif overall >= 6:
            level = "良好"
        elif overall >= 4:
            level = "一般"
        else:
            level = "需要改进"

        parts = [f"综合评级：{level}（{overall}/10分）"]

        if scores["faithfulness"] < 7:
            parts.append("忠实度偏低，建议检查优化后简历是否存在编造内容")
        if scores["jd_relevancy"] < 6:
            parts.append("与JD匹配度不足，建议进一步优化关键词")
        if scores["completeness"] < 6:
            parts.append("部分JD要求未被覆盖，建议补充相关经历")
        if scores["improvement"] >= 7:
            parts.append("简历优化效果显著")

        if len(parts) == 1:
            parts.append("各项指标表现均衡，简历质量较高")

        return "；".join(parts)

    @staticmethod
    def _parse_score(text: str) -> float:
        """从LLM输出中解析分数"""
        # 尝试提取数字
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text.strip())
        if numbers:
            score = float(numbers[0])
            return max(0, min(10, score))
        return 5.0  # 解析失败默认5分
