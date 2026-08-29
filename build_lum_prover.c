#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_SYMBOLS 64
#define MAX_RULES 64
#define MAX_BODY 8

// ==========================================================================
// FIRST-ORDER HORN CLAUSE DEDUCTION ENGINE
// ==========================================================================
typedef struct {
    char head[32];
    char body[MAX_BODY][32];
    int body_count;
} HornClause;

typedef struct {
    char facts[MAX_SYMBOLS][32];
    int fact_count;
    HornClause rules[MAX_RULES];
    int rule_count;
} KnowledgeBase;

static inline KnowledgeBase kb_create() {
    KnowledgeBase kb;
    kb.fact_count = 0;
    kb.rule_count = 0;
    return kb;
}

static inline void kb_assert_fact(KnowledgeBase *kb, const char *fact) {
    for (int i = 0; i < kb->fact_count; i++) {
        if (strcmp(kb->facts[i], fact) == 0) return;
    }
    strcpy(kb->facts[kb->fact_count++], fact);
}

static inline void kb_add_rule(KnowledgeBase *kb, const char *head, const char *body_csv) {
    HornClause *rule = &kb->rules[kb->rule_count++];
    strcpy(rule->head, head);
    rule->body_count = 0;

    char buffer[256];
    strcpy(buffer, body_csv);
    char *token = strtok(buffer, ", ");
    while (token != NULL && rule->body_count < MAX_BODY) {
        strcpy(rule->body[rule->body_count++], token);
        token = strtok(NULL, ", ");
    }
}

static inline bool kb_contains_fact(KnowledgeBase *kb, const char *fact) {
    for (int i = 0; i < kb->fact_count; i++) {
        if (strcmp(kb->facts[i], fact) == 0) return true;
    }
    return false;
}

// Forward Chaining Deductive Prover
static inline bool kb_prove(KnowledgeBase *kb, const char *query) {
    bool new_fact_inferred = true;
    while (new_fact_inferred) {
        new_fact_inferred = false;
        if (kb_contains_fact(kb, query)) return true;

        for (int i = 0; i < kb->rule_count; i++) {
            HornClause *r = &kb->rules[i];
            if (kb_contains_fact(kb, r->head)) continue;

            bool body_satisfied = true;
            for (int j = 0; j < r->body_count; j++) {
                if (!kb_contains_fact(kb, r->body[j])) {
                    body_satisfied = false;
                    break;
                }
            }

            if (body_satisfied) {
                kb_assert_fact(kb, r->head);
                new_fact_inferred = true;
            }
        }
    }
    return kb_contains_fact(kb, query);
}

int main() {
    printf("%s\n", "=== PURE FIRST-ORDER THEOREM PROVER (Horn Clause Inference) ===");
    KnowledgeBase kb = kb_create();
    kb_assert_fact(&kb, "topological_space");
    kb_assert_fact(&kb, "has_metric_tensor");
    kb_assert_fact(&kb, "symmetric_connection");
    kb_add_rule(&kb, "hausdorff", "topological_space");
    kb_add_rule(&kb, "riemannian_manifold", "hausdorff, has_metric_tensor");
    kb_add_rule(&kb, "levi_civita_space", "riemannian_manifold, symmetric_connection");
    printf("%s\n", "--- Query 1: Is the structure a Riemannian Manifold? ---");
    bool proof1 = kb_prove(&kb, "riemannian_manifold");
    printf("[Theorem Status]: %s\n", (proof1) ? "TRUE (Q.E.D.)" : "FALSE (Unproven)");
    printf("%s\n", "--- Query 2: Can Levi-Civita Connection be uniquely derived? ---");
    bool proof2 = kb_prove(&kb, "levi_civita_space");
    printf("[Theorem Status]: %s\n", (proof2) ? "TRUE (Q.E.D.)" : "FALSE (Unproven)");
    printf("%s\n", "--- Query 3: Is it a Flat Euclidean Space without curvature? ---");
    bool proof3 = kb_prove(&kb, "flat_euclidean");
    printf("[Theorem Status]: %s\n", (proof3) ? "TRUE (Q.E.D.)" : "FALSE (Unproven)");
    return 0;
}