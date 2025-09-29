"""
业务智能模块
实现风险等级划分、决策支持、成本效益分析和合规性检查功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class RiskLevelClassifier:
    """风险等级划分器"""
    
    def __init__(self):
        self.risk_levels = {
            'very_low': {'range': (0.0, 0.1), 'label': '极低风险', 'color': '#2E8B57'},
            'low': {'range': (0.1, 0.3), 'label': '低风险', 'color': '#32CD32'},
            'medium': {'range': (0.3, 0.6), 'label': '中等风险', 'color': '#FFD700'},
            'high': {'range': (0.6, 0.8), 'label': '高风险', 'color': '#FF8C00'},
            'very_high': {'range': (0.8, 1.0), 'label': '极高风险', 'color': '#DC143C'}
        }
        
        self.business_rules = {
            'credit_amount_threshold': 100000,  # 信贷金额阈值
            'income_ratio_threshold': 0.3,     # 收入比例阈值
            'credit_history_months': 24,       # 信用历史月数要求
            'debt_to_income_ratio': 0.4        # 债务收入比阈值
        }
    
    def classify_risk_level(self, risk_score: float, additional_factors: Dict = None) -> Dict:
        """
        根据风险评分和附加因素分类风险等级
        
        Args:
            risk_score: 模型预测的风险评分 (0-1)
            additional_factors: 附加业务因素
            
        Returns:
            风险等级分类结果
        """
        
        # 基础风险等级
        base_level = None
        for level, info in self.risk_levels.items():
            if info['range'][0] <= risk_score < info['range'][1]:
                base_level = level
                break
        
        if base_level is None:
            base_level = 'very_high' if risk_score >= 0.8 else 'very_low'
        
        # 考虑附加因素调整
        adjusted_level = self._adjust_risk_level(base_level, risk_score, additional_factors)
        
        result = {
            'risk_score': risk_score,
            'base_level': base_level,
            'adjusted_level': adjusted_level,
            'level_info': self.risk_levels[adjusted_level],
            'adjustment_reasons': self._get_adjustment_reasons(additional_factors),
            'business_recommendations': self._get_business_recommendations(adjusted_level, additional_factors)
        }
        
        return result
    
    def _adjust_risk_level(self, base_level: str, risk_score: float, factors: Dict) -> str:
        """根据业务因素调整风险等级"""
        
        if not factors:
            return base_level
        
        adjustment = 0
        
        # 信贷金额因素
        if 'credit_amount' in factors:
            amount = factors['credit_amount']
            if amount > self.business_rules['credit_amount_threshold']:
                adjustment += 1  # 大额贷款增加风险
        
        # 收入稳定性因素
        if 'income_stability' in factors:
            stability = factors['income_stability']
            if stability < 0.5:
                adjustment += 1  # 收入不稳定增加风险
            elif stability > 0.8:
                adjustment -= 1  # 收入稳定降低风险
        
        # 信用历史因素
        if 'credit_history_months' in factors:
            history = factors['credit_history_months']
            if history < self.business_rules['credit_history_months']:
                adjustment += 1  # 信用历史短增加风险
            elif history > 60:
                adjustment -= 1  # 长信用历史降低风险
        
        # 债务收入比因素
        if 'debt_to_income_ratio' in factors:
            ratio = factors['debt_to_income_ratio']
            if ratio > self.business_rules['debt_to_income_ratio']:
                adjustment += 1  # 高债务比增加风险
        
        # 应用调整
        levels = list(self.risk_levels.keys())
        current_index = levels.index(base_level)
        new_index = max(0, min(len(levels) - 1, current_index + adjustment))
        
        return levels[new_index]
    
    def _get_adjustment_reasons(self, factors: Dict) -> List[str]:
        """获取风险等级调整原因"""
        
        reasons = []
        if not factors:
            return reasons
        
        if factors.get('credit_amount', 0) > self.business_rules['credit_amount_threshold']:
            reasons.append("大额信贷申请增加风险等级")
        
        if factors.get('income_stability', 1) < 0.5:
            reasons.append("收入稳定性较低增加风险等级")
        elif factors.get('income_stability', 0) > 0.8:
            reasons.append("收入稳定性良好降低风险等级")
        
        if factors.get('credit_history_months', 100) < self.business_rules['credit_history_months']:
            reasons.append("信用历史较短增加风险等级")
        elif factors.get('credit_history_months', 0) > 60:
            reasons.append("信用历史良好降低风险等级")
        
        if factors.get('debt_to_income_ratio', 0) > self.business_rules['debt_to_income_ratio']:
            reasons.append("债务收入比过高增加风险等级")
        
        return reasons
    
    def _get_business_recommendations(self, risk_level: str, factors: Dict) -> List[str]:
        """获取业务建议"""
        
        recommendations = []
        
        if risk_level in ['very_low', 'low']:
            recommendations.extend([
                "建议批准信贷申请",
                "可提供优惠利率",
                "适合推荐其他金融产品"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "建议进一步审核",
                "可能需要额外担保或抵押",
                "建议降低信贷额度或提高利率"
            ])
        elif risk_level in ['high', 'very_high']:
            recommendations.extend([
                "建议拒绝信贷申请",
                "如批准需要严格的风控措施",
                "建议提供金融咨询服务"
            ])
        
        # 基于具体因素的建议
        if factors:
            if factors.get('credit_amount', 0) > self.business_rules['credit_amount_threshold']:
                recommendations.append("建议分期放款或降低信贷额度")
            
            if factors.get('income_stability', 1) < 0.5:
                recommendations.append("建议要求收入证明或担保人")
            
            if factors.get('debt_to_income_ratio', 0) > self.business_rules['debt_to_income_ratio']:
                recommendations.append("建议债务整合或延长还款期限")
        
        return recommendations

class DecisionSupportSystem:
    """决策支持系统"""
    
    def __init__(self):
        self.decision_rules = {
            'auto_approve_threshold': 0.1,    # 自动批准阈值
            'auto_reject_threshold': 0.8,     # 自动拒绝阈值
            'manual_review_range': (0.1, 0.8), # 人工审核范围
            'high_value_threshold': 500000,    # 高价值客户阈值
            'risk_tolerance': 0.05             # 风险容忍度
        }
        
        self.approval_matrix = {
            'very_low': {'auto_approve': True, 'conditions': []},
            'low': {'auto_approve': True, 'conditions': ['标准利率']},
            'medium': {'auto_approve': False, 'conditions': ['人工审核', '额外担保']},
            'high': {'auto_approve': False, 'conditions': ['严格审核', '高利率', '抵押要求']},
            'very_high': {'auto_approve': False, 'conditions': ['拒绝申请']}
        }
    
    def make_decision(self, risk_assessment: Dict, customer_profile: Dict = None) -> Dict:
        """
        基于风险评估和客户档案做出决策
        
        Args:
            risk_assessment: 风险评估结果
            customer_profile: 客户档案信息
            
        Returns:
            决策结果
        """
        
        risk_level = risk_assessment['adjusted_level']
        risk_score = risk_assessment['risk_score']
        
        # 基础决策
        base_decision = self._get_base_decision(risk_level, risk_score)
        
        # 考虑客户价值调整决策
        final_decision = self._adjust_decision_by_customer_value(
            base_decision, customer_profile, risk_assessment
        )
        
        # 生成决策解释
        explanation = self._generate_decision_explanation(
            final_decision, risk_assessment, customer_profile
        )
        
        result = {
            'decision': final_decision['action'],
            'confidence': final_decision['confidence'],
            'conditions': final_decision['conditions'],
            'recommended_amount': final_decision.get('recommended_amount'),
            'recommended_rate': final_decision.get('recommended_rate'),
            'explanation': explanation,
            'next_steps': self._get_next_steps(final_decision),
            'review_date': self._get_review_date(final_decision)
        }
        
        return result
    
    def _get_base_decision(self, risk_level: str, risk_score: float) -> Dict:
        """获取基础决策"""
        
        approval_info = self.approval_matrix[risk_level]
        
        if risk_score <= self.decision_rules['auto_approve_threshold']:
            action = 'auto_approve'
            confidence = 0.95
        elif risk_score >= self.decision_rules['auto_reject_threshold']:
            action = 'auto_reject'
            confidence = 0.95
        elif approval_info['auto_approve']:
            action = 'approve'
            confidence = 0.8
        else:
            action = 'manual_review'
            confidence = 0.6
        
        return {
            'action': action,
            'confidence': confidence,
            'conditions': approval_info['conditions'].copy(),
            'risk_level': risk_level,
            'risk_score': risk_score
        }
    
    def _adjust_decision_by_customer_value(self, base_decision: Dict, 
                                         customer_profile: Dict, 
                                         risk_assessment: Dict) -> Dict:
        """根据客户价值调整决策"""
        
        if not customer_profile:
            return base_decision
        
        # 计算客户价值评分
        customer_value = self._calculate_customer_value(customer_profile)
        
        # 高价值客户特殊处理
        if customer_value > 0.8 and base_decision['action'] in ['manual_review', 'auto_reject']:
            if base_decision['risk_score'] < 0.9:  # 不是极高风险
                base_decision['action'] = 'conditional_approve'
                base_decision['conditions'].extend(['高价值客户特殊审批', '额外监控'])
                base_decision['confidence'] = 0.7
        
        # 设置推荐金额和利率
        base_decision.update(self._calculate_loan_terms(customer_profile, risk_assessment))
        
        return base_decision
    
    def _calculate_customer_value(self, customer_profile: Dict) -> float:
        """计算客户价值评分"""
        
        value_score = 0.0
        
        # 收入水平 (30%)
        income = customer_profile.get('annual_income', 0)
        if income > 1000000:
            value_score += 0.3
        elif income > 500000:
            value_score += 0.2
        elif income > 200000:
            value_score += 0.1
        
        # 资产状况 (25%)
        assets = customer_profile.get('total_assets', 0)
        if assets > 5000000:
            value_score += 0.25
        elif assets > 2000000:
            value_score += 0.2
        elif assets > 1000000:
            value_score += 0.15
        elif assets > 500000:
            value_score += 0.1
        
        # 信用历史 (20%)
        credit_history = customer_profile.get('credit_history_years', 0)
        if credit_history > 10:
            value_score += 0.2
        elif credit_history > 5:
            value_score += 0.15
        elif credit_history > 2:
            value_score += 0.1
        
        # 银行关系 (15%)
        relationship_years = customer_profile.get('bank_relationship_years', 0)
        if relationship_years > 10:
            value_score += 0.15
        elif relationship_years > 5:
            value_score += 0.1
        elif relationship_years > 2:
            value_score += 0.05
        
        # 产品使用情况 (10%)
        products_count = customer_profile.get('products_count', 0)
        if products_count > 5:
            value_score += 0.1
        elif products_count > 3:
            value_score += 0.07
        elif products_count > 1:
            value_score += 0.05
        
        return min(1.0, value_score)
    
    def _calculate_loan_terms(self, customer_profile: Dict, risk_assessment: Dict) -> Dict:
        """计算推荐的贷款条件"""
        
        risk_score = risk_assessment['risk_score']
        requested_amount = customer_profile.get('requested_amount', 0)
        
        # 基础利率 (假设基准利率为4%)
        base_rate = 0.04
        
        # 根据风险调整利率
        risk_premium = risk_score * 0.08  # 最高8%的风险溢价
        recommended_rate = base_rate + risk_premium
        
        # 根据风险调整金额
        risk_factor = 1 - risk_score * 0.5  # 高风险最多减少50%
        recommended_amount = requested_amount * risk_factor
        
        # 客户价值调整
        customer_value = self._calculate_customer_value(customer_profile)
        if customer_value > 0.8:
            recommended_rate *= 0.9  # 高价值客户利率优惠
            recommended_amount = min(requested_amount, recommended_amount * 1.2)
        
        return {
            'recommended_amount': max(0, recommended_amount),
            'recommended_rate': recommended_rate,
            'max_amount': requested_amount,
            'min_rate': base_rate
        }
    
    def _generate_decision_explanation(self, decision: Dict, 
                                     risk_assessment: Dict, 
                                     customer_profile: Dict) -> str:
        """生成决策解释"""
        
        explanations = []
        
        # 风险因素解释
        risk_level = risk_assessment['adjusted_level']
        risk_score = risk_assessment['risk_score']
        
        explanations.append(f"客户风险等级为{self._get_risk_level_name(risk_level)}，风险评分为{risk_score:.3f}")
        
        # 决策原因
        action = decision['action']
        if action == 'auto_approve':
            explanations.append("风险评分极低，系统自动批准")
        elif action == 'auto_reject':
            explanations.append("风险评分极高，系统自动拒绝")
        elif action == 'approve':
            explanations.append("风险可控，建议批准申请")
        elif action == 'conditional_approve':
            explanations.append("考虑客户价值，条件性批准")
        elif action == 'manual_review':
            explanations.append("风险中等，需要人工审核")
        
        # 调整因素
        if risk_assessment.get('adjustment_reasons'):
            explanations.append("调整因素：" + "；".join(risk_assessment['adjustment_reasons']))
        
        # 客户价值因素
        if customer_profile:
            customer_value = self._calculate_customer_value(customer_profile)
            if customer_value > 0.8:
                explanations.append("客户为高价值客户，给予特殊考虑")
            elif customer_value > 0.6:
                explanations.append("客户价值较高，适当优惠")
        
        return "；".join(explanations)
    
    def _get_risk_level_name(self, risk_level: str) -> str:
        """获取风险等级中文名称"""
        
        names = {
            'very_low': '极低风险',
            'low': '低风险',
            'medium': '中等风险',
            'high': '高风险',
            'very_high': '极高风险'
        }
        return names.get(risk_level, risk_level)
    
    def _get_next_steps(self, decision: Dict) -> List[str]:
        """获取后续步骤"""
        
        action = decision['action']
        steps = []
        
        if action == 'auto_approve':
            steps = ['生成贷款合同', '安排放款流程', '设置还款提醒']
        elif action == 'approve':
            steps = ['最终审核确认', '生成贷款合同', '安排放款流程']
        elif action == 'conditional_approve':
            steps = ['验证附加条件', '高级审批', '生成特殊条款合同']
        elif action == 'manual_review':
            steps = ['分配审核员', '收集补充材料', '安排面谈']
        elif action == 'auto_reject':
            steps = ['发送拒绝通知', '提供改进建议', '记录拒绝原因']
        
        return steps
    
    def _get_review_date(self, decision: Dict) -> str:
        """获取复审日期"""
        
        action = decision['action']
        
        if action in ['auto_approve', 'approve']:
            # 批准后6个月复审
            review_date = datetime.now() + timedelta(days=180)
        elif action == 'conditional_approve':
            # 条件批准后3个月复审
            review_date = datetime.now() + timedelta(days=90)
        elif action == 'manual_review':
            # 人工审核后1个月复审
            review_date = datetime.now() + timedelta(days=30)
        else:
            # 拒绝后6个月可重新申请
            review_date = datetime.now() + timedelta(days=180)
        
        return review_date.strftime('%Y-%m-%d')

class CostBenefitAnalyzer:
    """成本效益分析器"""
    
    def __init__(self):
        self.cost_parameters = {
            'operational_cost_rate': 0.02,      # 运营成本率
            'risk_cost_rate': 0.05,             # 风险成本率
            'capital_cost_rate': 0.03,          # 资金成本率
            'processing_cost_per_loan': 500,    # 每笔贷款处理成本
            'monitoring_cost_rate': 0.01,       # 监控成本率
            'collection_cost_rate': 0.03        # 催收成本率
        }
        
        self.benefit_parameters = {
            'interest_margin_rate': 0.06,       # 利息收益率
            'fee_income_rate': 0.01,            # 手续费收入率
            'cross_selling_value': 2000,        # 交叉销售价值
            'customer_lifetime_value': 10000    # 客户生命周期价值
        }
    
    def analyze_portfolio_cost_benefit(self, loan_portfolio: List[Dict]) -> Dict:
        """
        分析贷款组合的成本效益
        
        Args:
            loan_portfolio: 贷款组合列表
            
        Returns:
            成本效益分析结果
        """
        
        total_amount = sum(loan['amount'] for loan in loan_portfolio)
        total_loans = len(loan_portfolio)
        
        # 计算总成本
        total_costs = self._calculate_total_costs(loan_portfolio, total_amount)
        
        # 计算总收益
        total_benefits = self._calculate_total_benefits(loan_portfolio, total_amount)
        
        # 计算净收益
        net_benefit = total_benefits - total_costs
        
        # 计算关键指标
        roi = (net_benefit / total_costs) if total_costs > 0 else 0
        profit_margin = (net_benefit / total_benefits) if total_benefits > 0 else 0
        
        # 风险调整收益
        avg_risk_score = np.mean([loan['risk_score'] for loan in loan_portfolio])
        risk_adjusted_return = net_benefit * (1 - avg_risk_score)
        
        result = {
            'portfolio_summary': {
                'total_loans': total_loans,
                'total_amount': total_amount,
                'average_loan_amount': total_amount / total_loans if total_loans > 0 else 0,
                'average_risk_score': avg_risk_score
            },
            'cost_analysis': total_costs,
            'benefit_analysis': total_benefits,
            'profitability_metrics': {
                'net_benefit': net_benefit,
                'roi': roi,
                'profit_margin': profit_margin,
                'risk_adjusted_return': risk_adjusted_return
            },
            'risk_metrics': self._calculate_risk_metrics(loan_portfolio),
            'recommendations': self._generate_portfolio_recommendations(
                net_benefit, roi, avg_risk_score
            )
        }
        
        return result
    
    def _calculate_total_costs(self, loan_portfolio: List[Dict], total_amount: float) -> Dict:
        """计算总成本"""
        
        costs = {
            'operational_costs': 0,
            'risk_costs': 0,
            'capital_costs': 0,
            'processing_costs': 0,
            'monitoring_costs': 0,
            'collection_costs': 0
        }
        
        for loan in loan_portfolio:
            amount = loan['amount']
            risk_score = loan['risk_score']
            
            # 运营成本
            costs['operational_costs'] += amount * self.cost_parameters['operational_cost_rate']
            
            # 风险成本（基于风险评分）
            costs['risk_costs'] += amount * self.cost_parameters['risk_cost_rate'] * risk_score
            
            # 资金成本
            costs['capital_costs'] += amount * self.cost_parameters['capital_cost_rate']
            
            # 处理成本
            costs['processing_costs'] += self.cost_parameters['processing_cost_per_loan']
            
            # 监控成本
            costs['monitoring_costs'] += amount * self.cost_parameters['monitoring_cost_rate']
            
            # 催收成本（高风险贷款）
            if risk_score > 0.6:
                costs['collection_costs'] += amount * self.cost_parameters['collection_cost_rate']
        
        costs['total_costs'] = sum(costs.values())
        
        return costs
    
    def _calculate_total_benefits(self, loan_portfolio: List[Dict], total_amount: float) -> Dict:
        """计算总收益"""
        
        benefits = {
            'interest_income': 0,
            'fee_income': 0,
            'cross_selling_income': 0,
            'customer_value_income': 0
        }
        
        for loan in loan_portfolio:
            amount = loan['amount']
            risk_score = loan['risk_score']
            customer_value = loan.get('customer_value', 0.5)
            
            # 利息收入（考虑风险调整）
            risk_adjusted_rate = self.benefit_parameters['interest_margin_rate'] * (1 + risk_score)
            benefits['interest_income'] += amount * risk_adjusted_rate
            
            # 手续费收入
            benefits['fee_income'] += amount * self.benefit_parameters['fee_income_rate']
            
            # 交叉销售收入（基于客户价值）
            benefits['cross_selling_income'] += (
                self.benefit_parameters['cross_selling_value'] * customer_value
            )
            
            # 客户生命周期价值
            benefits['customer_value_income'] += (
                self.benefit_parameters['customer_lifetime_value'] * customer_value * 0.1
            )  # 年化10%
        
        benefits['total_benefits'] = sum(benefits.values())
        
        return benefits
    
    def _calculate_risk_metrics(self, loan_portfolio: List[Dict]) -> Dict:
        """计算风险指标"""
        
        risk_scores = [loan['risk_score'] for loan in loan_portfolio]
        amounts = [loan['amount'] for loan in loan_portfolio]
        
        # 加权平均风险
        weighted_avg_risk = np.average(risk_scores, weights=amounts)
        
        # 风险分布
        risk_distribution = {
            'very_low': sum(1 for score in risk_scores if score < 0.1),
            'low': sum(1 for score in risk_scores if 0.1 <= score < 0.3),
            'medium': sum(1 for score in risk_scores if 0.3 <= score < 0.6),
            'high': sum(1 for score in risk_scores if 0.6 <= score < 0.8),
            'very_high': sum(1 for score in risk_scores if score >= 0.8)
        }
        
        # 预期损失
        expected_loss = sum(
            loan['amount'] * loan['risk_score'] * 0.5  # 假设违约损失率50%
            for loan in loan_portfolio
        )
        
        # 风险集中度
        risk_concentration = np.std(risk_scores)
        
        return {
            'weighted_average_risk': weighted_avg_risk,
            'risk_distribution': risk_distribution,
            'expected_loss': expected_loss,
            'risk_concentration': risk_concentration,
            'value_at_risk_95': np.percentile([
                loan['amount'] * loan['risk_score'] for loan in loan_portfolio
            ], 95)
        }
    
    def _generate_portfolio_recommendations(self, net_benefit: float, 
                                          roi: float, avg_risk: float) -> List[str]:
        """生成组合建议"""
        
        recommendations = []
        
        if net_benefit > 0:
            recommendations.append("✅ 组合整体盈利，建议继续执行")
        else:
            recommendations.append("❌ 组合亏损，需要调整策略")
        
        if roi > 0.15:
            recommendations.append("🎯 投资回报率优秀，可考虑扩大规模")
        elif roi > 0.08:
            recommendations.append("📈 投资回报率良好，保持当前策略")
        else:
            recommendations.append("⚠️ 投资回报率偏低，需要优化")
        
        if avg_risk > 0.6:
            recommendations.append("🔴 平均风险较高，建议加强风控")
        elif avg_risk > 0.4:
            recommendations.append("🟡 平均风险中等，需要平衡收益与风险")
        else:
            recommendations.append("🟢 平均风险较低，可适当提高收益目标")
        
        return recommendations

class ComplianceChecker:
    """合规性检查器"""
    
    def __init__(self):
        self.regulatory_requirements = {
            'max_loan_to_income_ratio': 0.5,     # 最大贷款收入比
            'min_credit_score': 600,             # 最低信用评分
            'max_debt_to_income_ratio': 0.43,    # 最大债务收入比
            'min_down_payment_ratio': 0.2,       # 最低首付比例
            'max_loan_amount': 5000000,          # 最大贷款金额
            'min_income_verification': True,      # 收入验证要求
            'kyc_requirements': True,            # KYC要求
            'aml_screening': True                # 反洗钱筛查
        }
        
        self.compliance_rules = {
            'fair_lending': {
                'protected_classes': ['race', 'gender', 'age', 'religion'],
                'prohibited_factors': ['marital_status', 'family_status']
            },
            'data_privacy': {
                'data_retention_days': 2555,  # 7年
                'consent_required': True,
                'data_minimization': True
            },
            'risk_management': {
                'stress_testing_required': True,
                'capital_adequacy_ratio': 0.08,
                'liquidity_ratio': 0.03
            }
        }
    
    def check_loan_compliance(self, loan_application: Dict) -> Dict:
        """
        检查单笔贷款的合规性
        
        Args:
            loan_application: 贷款申请信息
            
        Returns:
            合规性检查结果
        """
        
        compliance_results = {
            'overall_compliant': True,
            'violations': [],
            'warnings': [],
            'requirements_met': [],
            'recommendations': []
        }
        
        # 基础监管要求检查
        self._check_basic_requirements(loan_application, compliance_results)
        
        # 公平放贷检查
        self._check_fair_lending(loan_application, compliance_results)
        
        # 数据隐私检查
        self._check_data_privacy(loan_application, compliance_results)
        
        # KYC/AML检查
        self._check_kyc_aml(loan_application, compliance_results)
        
        # 更新总体合规状态
        compliance_results['overall_compliant'] = len(compliance_results['violations']) == 0
        
        # 生成合规报告
        compliance_results['compliance_score'] = self._calculate_compliance_score(compliance_results)
        compliance_results['next_actions'] = self._get_compliance_actions(compliance_results)
        
        return compliance_results
    
    def _check_basic_requirements(self, application: Dict, results: Dict):
        """检查基础监管要求"""
        
        # 贷款收入比检查
        loan_amount = application.get('loan_amount', 0)
        annual_income = application.get('annual_income', 0)
        
        if annual_income > 0:
            loan_to_income = loan_amount / annual_income
            if loan_to_income > self.regulatory_requirements['max_loan_to_income_ratio']:
                results['violations'].append(
                    f"贷款收入比{loan_to_income:.2f}超过监管要求{self.regulatory_requirements['max_loan_to_income_ratio']}"
                )
            else:
                results['requirements_met'].append("贷款收入比符合要求")
        
        # 信用评分检查
        credit_score = application.get('credit_score', 0)
        if credit_score < self.regulatory_requirements['min_credit_score']:
            results['violations'].append(
                f"信用评分{credit_score}低于最低要求{self.regulatory_requirements['min_credit_score']}"
            )
        else:
            results['requirements_met'].append("信用评分符合要求")
        
        # 债务收入比检查
        total_debt = application.get('total_debt', 0)
        if annual_income > 0:
            debt_to_income = total_debt / annual_income
            if debt_to_income > self.regulatory_requirements['max_debt_to_income_ratio']:
                results['violations'].append(
                    f"债务收入比{debt_to_income:.2f}超过监管要求{self.regulatory_requirements['max_debt_to_income_ratio']}"
                )
            else:
                results['requirements_met'].append("债务收入比符合要求")
        
        # 贷款金额检查
        if loan_amount > self.regulatory_requirements['max_loan_amount']:
            results['violations'].append(
                f"贷款金额{loan_amount}超过最大限额{self.regulatory_requirements['max_loan_amount']}"
            )
        else:
            results['requirements_met'].append("贷款金额符合要求")
        
        # 收入验证检查
        if self.regulatory_requirements['min_income_verification']:
            if not application.get('income_verified', False):
                results['violations'].append("缺少收入验证文件")
            else:
                results['requirements_met'].append("收入验证完成")
    
    def _check_fair_lending(self, application: Dict, results: Dict):
        """检查公平放贷合规性"""
        
        # 检查是否使用了禁止的因素
        prohibited_factors = self.compliance_rules['fair_lending']['prohibited_factors']
        
        for factor in prohibited_factors:
            if factor in application:
                results['warnings'].append(f"使用了可能涉及歧视的因素: {factor}")
        
        # 检查决策过程的公平性
        if 'decision_factors' in application:
            decision_factors = application['decision_factors']
            protected_classes = self.compliance_rules['fair_lending']['protected_classes']
            
            for factor in decision_factors:
                if any(pc in factor.lower() for pc in protected_classes):
                    results['violations'].append(f"决策因素可能涉及受保护类别: {factor}")
        
        results['requirements_met'].append("公平放贷检查完成")
    
    def _check_data_privacy(self, application: Dict, results: Dict):
        """检查数据隐私合规性"""
        
        privacy_rules = self.compliance_rules['data_privacy']
        
        # 检查用户同意
        if privacy_rules['consent_required']:
            if not application.get('privacy_consent', False):
                results['violations'].append("缺少用户隐私同意")
            else:
                results['requirements_met'].append("用户隐私同意已获取")
        
        # 检查数据最小化
        if privacy_rules['data_minimization']:
            collected_fields = len(application.keys())
            if collected_fields > 20:  # 假设合理字段数为20
                results['warnings'].append("收集的数据字段可能过多，建议检查数据最小化原则")
            else:
                results['requirements_met'].append("数据收集符合最小化原则")
        
        # 检查数据保留期限
        if 'data_collection_date' in application:
            collection_date = datetime.strptime(application['data_collection_date'], '%Y-%m-%d')
            retention_days = privacy_rules['data_retention_days']
            expiry_date = collection_date + timedelta(days=retention_days)
            
            if datetime.now() > expiry_date:
                results['violations'].append("数据保留期限已超过法定要求")
            else:
                results['requirements_met'].append("数据保留期限符合要求")
    
    def _check_kyc_aml(self, application: Dict, results: Dict):
        """检查KYC/AML合规性"""
        
        # KYC检查
        if self.regulatory_requirements['kyc_requirements']:
            required_kyc_docs = ['identity_verified', 'address_verified', 'income_verified']
            
            for doc in required_kyc_docs:
                if not application.get(doc, False):
                    results['violations'].append(f"缺少KYC文件: {doc}")
                else:
                    results['requirements_met'].append(f"KYC文件已验证: {doc}")
        
        # AML检查
        if self.regulatory_requirements['aml_screening']:
            if not application.get('aml_screening_passed', False):
                results['violations'].append("未通过反洗钱筛查")
            else:
                results['requirements_met'].append("反洗钱筛查通过")
            
            # 检查可疑交易模式
            if application.get('large_cash_transactions', False):
                results['warnings'].append("存在大额现金交易，需要额外关注")
            
            if application.get('frequent_transactions', False):
                results['warnings'].append("存在频繁交易模式，需要监控")
    
    def _calculate_compliance_score(self, results: Dict) -> float:
        """计算合规评分"""
        
        total_checks = (len(results['violations']) + 
                       len(results['warnings']) + 
                       len(results['requirements_met']))
        
        if total_checks == 0:
            return 1.0
        
        # 违规扣分更多，警告扣分较少
        violation_penalty = len(results['violations']) * 0.2
        warning_penalty = len(results['warnings']) * 0.05
        
        score = 1.0 - (violation_penalty + warning_penalty) / total_checks
        return max(0.0, min(1.0, score))
    
    def _get_compliance_actions(self, results: Dict) -> List[str]:
        """获取合规行动建议"""
        
        actions = []
        
        if results['violations']:
            actions.append("🔴 立即处理合规违规问题")
            actions.append("📋 更新合规检查清单")
            actions.append("🔍 加强内部审核流程")
        
        if results['warnings']:
            actions.append("🟡 关注潜在合规风险")
            actions.append("📊 定期监控相关指标")
        
        if results['overall_compliant']:
            actions.append("✅ 继续保持合规标准")
            actions.append("📈 定期更新合规要求")
        
        return actions

class BusinessIntelligenceSystem:
    """业务智能系统集成"""
    
    def __init__(self, output_dir: str = "business_intelligence_output"):
        self.output_dir = output_dir
        self.reports_dir = os.path.join(output_dir, "reports")
        self.charts_dir = os.path.join(output_dir, "charts")
        self.data_dir = os.path.join(output_dir, "data")
        
        # 创建输出目录
        for directory in [self.output_dir, self.reports_dir, self.charts_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 初始化各个组件
        self.risk_classifier = RiskLevelClassifier()
        self.decision_system = DecisionSupportSystem()
        self.cost_benefit_analyzer = CostBenefitAnalyzer()
        self.compliance_checker = ComplianceChecker()
    
    def comprehensive_business_analysis(self, loan_applications: List[Dict]) -> Dict:
        """
        综合业务分析
        
        Args:
            loan_applications: 贷款申请列表
            
        Returns:
            综合分析结果
        """
        
        print("🚀 开始综合业务分析...")
        
        results = {
            'analysis_summary': {
                'total_applications': len(loan_applications),
                'analysis_date': datetime.now().isoformat(),
                'analysis_scope': '风险分级、决策支持、成本效益、合规检查'
            },
            'risk_analysis': {},
            'decision_analysis': {},
            'cost_benefit_analysis': {},
            'compliance_analysis': {},
            'business_recommendations': []
        }
        
        # 1. 风险分析
        print("📊 执行风险等级分析...")
        results['risk_analysis'] = self._analyze_risk_levels(loan_applications)
        
        # 2. 决策分析
        print("🎯 执行决策支持分析...")
        results['decision_analysis'] = self._analyze_decisions(loan_applications)
        
        # 3. 成本效益分析
        print("💰 执行成本效益分析...")
        results['cost_benefit_analysis'] = self._analyze_cost_benefit(loan_applications)
        
        # 4. 合规性分析
        print("⚖️ 执行合规性检查...")
        results['compliance_analysis'] = self._analyze_compliance(loan_applications)
        
        # 5. 生成业务建议
        print("💡 生成业务建议...")
        results['business_recommendations'] = self._generate_business_recommendations(results)
        
        # 6. 生成图表
        print("📈 生成分析图表...")
        self._generate_analysis_charts(results)
        
        print("✅ 综合业务分析完成")
        return results
    
    def _analyze_risk_levels(self, applications: List[Dict]) -> Dict:
        """分析风险等级分布"""
        
        risk_results = []
        risk_distribution = {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
        for app in applications:
            risk_score = app.get('risk_score', 0.5)
            additional_factors = {
                'credit_amount': app.get('loan_amount', 0),
                'income_stability': app.get('income_stability', 0.7),
                'credit_history_months': app.get('credit_history_months', 36),
                'debt_to_income_ratio': app.get('debt_to_income_ratio', 0.3)
            }
            
            risk_result = self.risk_classifier.classify_risk_level(risk_score, additional_factors)
            risk_results.append(risk_result)
            risk_distribution[risk_result['adjusted_level']] += 1
        
        return {
            'individual_results': risk_results,
            'distribution': risk_distribution,
            'statistics': {
                'average_risk_score': np.mean([r['risk_score'] for r in risk_results]),
                'high_risk_percentage': (risk_distribution['high'] + risk_distribution['very_high']) / len(applications),
                'low_risk_percentage': (risk_distribution['very_low'] + risk_distribution['low']) / len(applications)
            }
        }
    
    def _analyze_decisions(self, applications: List[Dict]) -> Dict:
        """分析决策结果"""
        
        decision_results = []
        decision_distribution = {'auto_approve': 0, 'approve': 0, 'conditional_approve': 0, 
                               'manual_review': 0, 'auto_reject': 0}
        
        for app in applications:
            # 构建风险评估结果
            risk_assessment = {
                'risk_score': app.get('risk_score', 0.5),
                'adjusted_level': app.get('risk_level', 'medium')
            }
            
            # 构建客户档案
            customer_profile = {
                'annual_income': app.get('annual_income', 500000),
                'total_assets': app.get('total_assets', 1000000),
                'credit_history_years': app.get('credit_history_years', 5),
                'bank_relationship_years': app.get('bank_relationship_years', 3),
                'products_count': app.get('products_count', 2),
                'requested_amount': app.get('loan_amount', 100000)
            }
            
            decision_result = self.decision_system.make_decision(risk_assessment, customer_profile)
            decision_results.append(decision_result)
            decision_distribution[decision_result['decision']] += 1
        
        return {
            'individual_results': decision_results,
            'distribution': decision_distribution,
            'statistics': {
                'approval_rate': (decision_distribution['auto_approve'] + decision_distribution['approve'] + 
                                decision_distribution['conditional_approve']) / len(applications),
                'rejection_rate': decision_distribution['auto_reject'] / len(applications),
                'manual_review_rate': decision_distribution['manual_review'] / len(applications),
                'average_confidence': np.mean([r['confidence'] for r in decision_results])
            }
        }
    
    def _analyze_cost_benefit(self, applications: List[Dict]) -> Dict:
        """分析成本效益"""
        
        # 构建贷款组合
        loan_portfolio = []
        for app in applications:
            loan_portfolio.append({
                'amount': app.get('loan_amount', 100000),
                'risk_score': app.get('risk_score', 0.5),
                'customer_value': app.get('customer_value', 0.6)
            })
        
        return self.cost_benefit_analyzer.analyze_portfolio_cost_benefit(loan_portfolio)
    
    def _analyze_compliance(self, applications: List[Dict]) -> Dict:
        """分析合规性"""
        
        compliance_results = []
        compliance_stats = {
            'compliant_count': 0,
            'violation_count': 0,
            'warning_count': 0,
            'average_compliance_score': 0
        }
        
        for app in applications:
            compliance_result = self.compliance_checker.check_loan_compliance(app)
            compliance_results.append(compliance_result)
            
            if compliance_result['overall_compliant']:
                compliance_stats['compliant_count'] += 1
            
            compliance_stats['violation_count'] += len(compliance_result['violations'])
            compliance_stats['warning_count'] += len(compliance_result['warnings'])
        
        if compliance_results:
            compliance_stats['average_compliance_score'] = np.mean([
                r['compliance_score'] for r in compliance_results
            ])
        
        compliance_stats['compliance_rate'] = compliance_stats['compliant_count'] / len(applications)
        
        return {
            'individual_results': compliance_results,
            'statistics': compliance_stats,
            'common_violations': self._get_common_violations(compliance_results),
            'improvement_areas': self._get_improvement_areas(compliance_results)
        }
    
    def _get_common_violations(self, compliance_results: List[Dict]) -> Dict:
        """获取常见违规问题"""
        
        violation_counts = {}
        for result in compliance_results:
            for violation in result['violations']:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        # 返回前5个最常见的违规
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_violations[:5])
    
    def _get_improvement_areas(self, compliance_results: List[Dict]) -> List[str]:
        """获取改进建议"""
        
        improvement_areas = []
        
        # 统计各类问题
        total_violations = sum(len(r['violations']) for r in compliance_results)
        total_warnings = sum(len(r['warnings']) for r in compliance_results)
        
        if total_violations > len(compliance_results) * 0.1:
            improvement_areas.append("加强基础合规检查流程")
        
        if total_warnings > len(compliance_results) * 0.2:
            improvement_areas.append("完善风险预警机制")
        
        # 检查合规评分分布
        scores = [r['compliance_score'] for r in compliance_results]
        if np.mean(scores) < 0.8:
            improvement_areas.append("提升整体合规水平")
        
        if np.std(scores) > 0.2:
            improvement_areas.append("标准化合规检查标准")
        
        return improvement_areas
    
    def _generate_business_recommendations(self, results: Dict) -> List[str]:
        """生成业务建议"""
        
        recommendations = []
        
        # 基于风险分析的建议
        risk_stats = results['risk_analysis']['statistics']
        if risk_stats['high_risk_percentage'] > 0.3:
            recommendations.append("🔴 高风险申请比例过高，建议加强风控标准")
        elif risk_stats['high_risk_percentage'] < 0.1:
            recommendations.append("🟢 风险控制良好，可考虑适当放宽标准以提高业务量")
        
        # 基于决策分析的建议
        decision_stats = results['decision_analysis']['statistics']
        if decision_stats['manual_review_rate'] > 0.4:
            recommendations.append("⚠️ 人工审核比例过高，建议优化自动化决策规则")
        
        if decision_stats['approval_rate'] < 0.3:
            recommendations.append("📉 批准率较低，可能影响业务增长，建议评估标准")
        elif decision_stats['approval_rate'] > 0.8:
            recommendations.append("📈 批准率较高，建议关注风险控制")
        
        # 基于成本效益分析的建议
        profitability = results['cost_benefit_analysis']['profitability_metrics']
        if profitability['roi'] < 0.1:
            recommendations.append("💰 投资回报率偏低，建议优化成本结构或提高收益")
        
        if profitability['net_benefit'] < 0:
            recommendations.append("❌ 组合亏损，需要立即调整业务策略")
        
        # 基于合规分析的建议
        compliance_stats = results['compliance_analysis']['statistics']
        if compliance_stats['compliance_rate'] < 0.9:
            recommendations.append("⚖️ 合规率偏低，需要加强合规培训和流程管控")
        
        if compliance_stats['violation_count'] > 0:
            recommendations.append("🚨 存在合规违规，需要立即整改")
        
        return recommendations
    
    def _generate_analysis_charts(self, results: Dict):
        """生成分析图表"""
        
        # 1. 风险等级分布饼图
        self._create_risk_distribution_chart(results['risk_analysis'])
        
        # 2. 决策结果分布图
        self._create_decision_distribution_chart(results['decision_analysis'])
        
        # 3. 成本效益分析图
        self._create_cost_benefit_chart(results['cost_benefit_analysis'])
        
        # 4. 合规性分析图
        self._create_compliance_chart(results['compliance_analysis'])
        
        # 5. 综合仪表板
        self._create_business_dashboard(results)
    
    def _create_risk_distribution_chart(self, risk_analysis: Dict):
        """创建风险分布图表"""
        
        distribution = risk_analysis['distribution']
        
        # 饼图
        fig = go.Figure(data=[go.Pie(
            labels=[f"{level}({count})" for level, count in distribution.items()],
            values=list(distribution.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="风险等级分布",
            annotations=[dict(text='风险分布', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        chart_path = os.path.join(self.charts_dir, "risk_distribution.html")
        fig.write_html(chart_path)
    
    def _create_decision_distribution_chart(self, decision_analysis: Dict):
        """创建决策分布图表"""
        
        distribution = decision_analysis['distribution']
        
        # 柱状图
        fig = go.Figure(data=[
            go.Bar(x=list(distribution.keys()), y=list(distribution.values()))
        ])
        
        fig.update_layout(
            title="决策结果分布",
            xaxis_title="决策类型",
            yaxis_title="数量"
        )
        
        chart_path = os.path.join(self.charts_dir, "decision_distribution.html")
        fig.write_html(chart_path)
    
    def _create_cost_benefit_chart(self, cost_benefit_analysis: Dict):
        """创建成本效益图表"""
        
        costs = cost_benefit_analysis['cost_analysis']
        benefits = cost_benefit_analysis['benefit_analysis']
        
        # 成本收益对比图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('成本构成', '收益构成'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # 成本饼图
        fig.add_trace(go.Pie(
            labels=list(costs.keys())[:-1],  # 排除total_costs
            values=list(costs.values())[:-1],
            name="成本"
        ), row=1, col=1)
        
        # 收益饼图
        fig.add_trace(go.Pie(
            labels=list(benefits.keys())[:-1],  # 排除total_benefits
            values=list(benefits.values())[:-1],
            name="收益"
        ), row=1, col=2)
        
        fig.update_layout(title_text="成本效益分析")
        
        chart_path = os.path.join(self.charts_dir, "cost_benefit_analysis.html")
        fig.write_html(chart_path)
    
    def _create_compliance_chart(self, compliance_analysis: Dict):
        """创建合规性图表"""
        
        stats = compliance_analysis['statistics']
        
        # 合规指标仪表盘
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = stats['compliance_rate'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "合规率 (%)"},
            delta = {'reference': 95},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        
        fig.update_layout(title="合规性仪表盘")
        
        chart_path = os.path.join(self.charts_dir, "compliance_dashboard.html")
        fig.write_html(chart_path)
    
    def _create_business_dashboard(self, results: Dict):
        """创建业务综合仪表板"""
        
        # 提取关键指标
        risk_stats = results['risk_analysis']['statistics']
        decision_stats = results['decision_analysis']['statistics']
        profitability = results['cost_benefit_analysis']['profitability_metrics']
        compliance_stats = results['compliance_analysis']['statistics']
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('风险指标', '决策指标', '盈利指标', '合规指标'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # 风险指标
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = risk_stats['high_risk_percentage'] * 100,
            title = {'text': "高风险比例 (%)"},
            gauge = {'axis': {'range': [0, 50]},
                    'bar': {'color': "red"},
                    'steps': [{'range': [0, 20], 'color': "lightgreen"},
                             {'range': [20, 35], 'color': "yellow"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 30}}
        ), row=1, col=1)
        
        # 决策指标
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = decision_stats['approval_rate'] * 100,
            title = {'text': "批准率 (%)"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [{'range': [0, 40], 'color': "lightgray"},
                             {'range': [40, 80], 'color': "lightblue"}],
                    'threshold': {'line': {'color': "green", 'width': 4},
                                'thickness': 0.75, 'value': 15}}
        ), row=2, col=1)
        
        # 合规指标
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = compliance_stats['compliance_rate'] * 100,
            title = {'text': "合规率 (%)"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [{'range': [0, 80], 'color': "lightgray"},
                             {'range': [80, 95], 'color': "lightpurple"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 95}}
        ), row=2, col=2)
        
        fig.update_layout(
            title="业务综合仪表板",
            height=600
        )
        
        chart_path = os.path.join(self.charts_dir, "business_dashboard.html")
        fig.write_html(chart_path)
    
    def generate_business_report(self, analysis_results: Dict) -> str:
        """生成业务分析报告"""
        
        report_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>业务智能分析报告</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 40px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
                .recommendation {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745; }}
                .warning {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                .danger {{ background: #f8d7da; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #dc3545; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .status-good {{ color: #28a745; font-weight: bold; }}
                .status-warning {{ color: #ffc107; font-weight: bold; }}
                .status-danger {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏦 业务智能分析报告</h1>
                    <p>信贷风险管理与决策支持系统</p>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>📊 分析概览</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{analysis_results['analysis_summary']['total_applications']}</div>
                            <div class="metric-label">总申请数量</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{analysis_results['decision_analysis']['statistics']['approval_rate']:.1%}</div>
                            <div class="metric-label">批准率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{analysis_results['risk_analysis']['statistics']['high_risk_percentage']:.1%}</div>
                            <div class="metric-label">高风险比例</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{analysis_results['compliance_analysis']['statistics']['compliance_rate']:.1%}</div>
                            <div class="metric-label">合规率</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 风险等级分析</h2>
                    <p>基于模型评分和业务规则的综合风险评估结果：</p>
                    <table>
                        <tr><th>风险等级</th><th>数量</th><th>占比</th><th>状态</th></tr>
        """
        
        # 添加风险分布表格
        risk_dist = analysis_results['risk_analysis']['distribution']
        total_apps = analysis_results['analysis_summary']['total_applications']
        
        risk_levels = {
            'very_low': '极低风险',
            'low': '低风险', 
            'medium': '中等风险',
            'high': '高风险',
            'very_high': '极高风险'
        }
        
        for level, count in risk_dist.items():
            percentage = count / total_apps if total_apps > 0 else 0
            status_class = 'status-good' if level in ['very_low', 'low'] else 'status-warning' if level == 'medium' else 'status-danger'
            report_content += f"""
                        <tr>
                            <td>{risk_levels[level]}</td>
                            <td>{count}</td>
                            <td>{percentage:.1%}</td>
                            <td class="{status_class}">{'正常' if level in ['very_low', 'low'] else '关注' if level == 'medium' else '警告'}</td>
                        </tr>
            """
        
        report_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>💼 决策支持分析</h2>
                    <p>自动化决策系统的执行结果和效率分析：</p>
        """
        
        # 添加决策统计
        decision_stats = analysis_results['decision_analysis']['statistics']
        report_content += f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{decision_stats['approval_rate']:.1%}</div>
                            <div class="metric-label">总批准率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{decision_stats['rejection_rate']:.1%}</div>
                            <div class="metric-label">拒绝率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{decision_stats['manual_review_rate']:.1%}</div>
                            <div class="metric-label">人工审核率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{decision_stats['average_confidence']:.1%}</div>
                            <div class="metric-label">平均置信度</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>💰 成本效益分析</h2>
                    <p>贷款组合的财务表现和盈利能力评估：</p>
        """
        
        # 添加成本效益指标
        profitability = analysis_results['cost_benefit_analysis']['profitability_metrics']
        portfolio = analysis_results['cost_benefit_analysis']['portfolio_summary']
        
        report_content += f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">¥{profitability['net_benefit']:,.0f}</div>
                            <div class="metric-label">净收益</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{profitability['roi']:.1%}</div>
                            <div class="metric-label">投资回报率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{profitability['profit_margin']:.1%}</div>
                            <div class="metric-label">利润率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">¥{portfolio['total_amount']:,.0f}</div>
                            <div class="metric-label">总贷款金额</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚖️ 合规性检查</h2>
                    <p>监管要求遵循情况和风险控制评估：</p>
        """
        
        # 添加合规统计
        compliance_stats = analysis_results['compliance_analysis']['statistics']
        report_content += f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{compliance_stats['compliance_rate']:.1%}</div>
                            <div class="metric-label">合规率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{compliance_stats['violation_count']}</div>
                            <div class="metric-label">违规数量</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{compliance_stats['warning_count']}</div>
                            <div class="metric-label">警告数量</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{compliance_stats['average_compliance_score']:.1%}</div>
                            <div class="metric-label">平均合规评分</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 业务建议</h2>
                    <p>基于分析结果的具体改进建议：</p>
        """
        
        # 添加业务建议
        for recommendation in analysis_results['business_recommendations']:
            if '🔴' in recommendation or '❌' in recommendation or '🚨' in recommendation:
                report_content += f'<div class="danger">{recommendation}</div>'
            elif '⚠️' in recommendation or '🟡' in recommendation:
                report_content += f'<div class="warning">{recommendation}</div>'
            else:
                report_content += f'<div class="recommendation">{recommendation}</div>'
        
        report_content += """
                </div>
                
                <div class="section">
                    <h2>📈 图表分析</h2>
                    <p>详细的可视化分析图表已生成，请查看以下文件：</p>
                    <ul>
                        <li><a href="charts/risk_distribution.html">风险等级分布图</a></li>
                        <li><a href="charts/decision_distribution.html">决策结果分布图</a></li>
                        <li><a href="charts/cost_benefit_analysis.html">成本效益分析图</a></li>
                        <li><a href="charts/compliance_dashboard.html">合规性仪表板</a></li>
                        <li><a href="charts/business_dashboard.html">业务综合仪表板</a></li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>📋 总结</h2>
                    <p>本次业务智能分析涵盖了风险管理、决策支持、成本效益和合规性四个核心维度。
                    通过系统化的分析，为信贷业务的优化提供了数据支持和决策依据。
                    建议定期执行此类分析，持续监控业务表现，及时调整策略。</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        report_path = os.path.join(self.reports_dir, "business_intelligence_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path

# 示例使用
if __name__ == "__main__":
    # 创建业务智能系统
    bi_system = BusinessIntelligenceSystem()
    
    # 示例贷款申请数据
    sample_applications = [
        {
            'loan_amount': 200000,
            'annual_income': 600000,
            'risk_score': 0.2,
            'credit_score': 750,
            'total_debt': 100000,
            'income_verified': True,
            'privacy_consent': True,
            'aml_screening_passed': True,
            'customer_value': 0.8
        },
        {
            'loan_amount': 500000,
            'annual_income': 800000,
            'risk_score': 0.6,
            'credit_score': 650,
            'total_debt': 300000,
            'income_verified': True,
            'privacy_consent': True,
            'aml_screening_passed': True,
            'customer_value': 0.6
        },
        {
            'loan_amount': 100000,
            'annual_income': 400000,
            'risk_score': 0.1,
            'credit_score': 800,
            'total_debt': 50000,
            'income_verified': True,
            'privacy_consent': True,
            'aml_screening_passed': True,
            'customer_value': 0.7
        }
    ]
    
    # 执行综合分析
    results = bi_system.comprehensive_business_analysis(sample_applications)
    
    # 生成报告
    report_path = bi_system.generate_business_report(results)
    
    print(f"✅ 业务智能分析完成！")
    print(f"📊 报告已生成: {report_path}")
    print(f"📈 图表目录: {bi_system.charts_dir}")
    print(f"📁 输出目录: {bi_system.output_dir}")