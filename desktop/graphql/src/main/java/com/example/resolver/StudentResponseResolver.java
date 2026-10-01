package com.example.resolver;
import java.util.Base64;

import org.springframework.stereotype.Service;

import com.coxautodev.graphql.tools.GraphQLResolver;
import com.example.resolver.SubjectConnection;

@Service
public class StudentResponseResolver implements GraphQLResolver<StudentResponse> {

    public SubjectConnection getLearningSubjects(
            StudentResponse studentResponse,
            SubjectNameFilter subjectNameFilter,
            Integer first,
            String after) {

        List<Subject> allSubjects = studentResponse.getStudent().getLearningSubjects();
        List<SubjectResponse> filtered = new ArrayList<>();

        // Filter by subject name
        if (allSubjects != null) {
            for (Subject subject : allSubjects) {
                if (subjectNameFilter == null ||
                        subjectNameFilter.name().equalsIgnoreCase(subject.getSubjectName())) {
                    filtered.add(new SubjectResponse(subject));
                }
            }
        }

        // Handle cursor
        int startIndex = 0;
        if (after != null) {
            // Decode base64 cursor
            String decoded = new String(Base64.getDecoder().decode(after));
            startIndex = Integer.parseInt(decoded) + 1; // move to next item
        }

        // Paginate
        int endIndex = Math.min(startIndex + first, filtered.size());
        List<SubjectResponse> pagedSubjects = filtered.subList(startIndex, endIndex);

        // Build edges with cursor
        List<SubjectEdge> edges = new ArrayList<>();
        for (int i = 0; i < pagedSubjects.size(); i++) {
            int cursorIndex = startIndex + i;
            String cursor = Base64.getEncoder().encodeToString(String.valueOf(cursorIndex).getBytes());
            edges.add(new SubjectEdge(pagedSubjects.get(i), cursor));
        }

        // PageInfo
        boolean hasNextPage = endIndex < filtered.size();
        String endCursor = edges.isEmpty() ? null : edges.get(edges.size() - 1).getCursor();

        PageInfo pageInfo = new PageInfo(hasNextPage, endCursor);
        return new SubjectConnection(edges, pageInfo);
    }
}