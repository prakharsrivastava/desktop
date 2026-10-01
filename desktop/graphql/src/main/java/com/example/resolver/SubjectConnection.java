package com.example.resolver;

public class SubjectConnection {
    private List<SubjectEdge> edges;
    private PageInfo pageInfo;

    public SubjectConnection(List<SubjectEdge> edges, PageInfo pageInfo) {
        this.edges = edges;
        this.pageInfo = pageInfo;
    }
    // getters & setters
}

public class SubjectEdge {
    private SubjectResponse node;
    private String cursor;

    public SubjectEdge(SubjectResponse node, String cursor) {
        this.node = node;
        this.cursor = cursor;
    }
    // getters & setters
}

public class PageInfo {
    private boolean hasNextPage;
    private String endCursor;

    public PageInfo(boolean hasNextPage, String endCursor) {
        this.hasNextPage = hasNextPage;
        this.endCursor = endCursor;
    }
    // getters & setters
}