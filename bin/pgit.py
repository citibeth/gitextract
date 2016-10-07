import pygit2
import os
from collections import defaultdict
import sys
import datetime
import collections
import re

def commit2str(commit):
    stime = datetime.datetime.utcfromtimestamp(commit.commit_time).strftime('%Y-%m-%dT%H:%M:%SZ')
    return '{} {} {}'.format(str(commit.id)[:6], stime, commit.author.email)
#    return '{} {} {}'.format(str(commit.id)[:6], stime, commit.message)


def all_refs_prefix(repo, prefix='refs/remotes/origin'):
    """Enumerates all refs in the repo starting with a particular prefix."""
    for ref_str in repo.listall_references():
        if ref_str.startswith(prefix):
            yield repo.lookup_reference(ref_str)

def refs2commits(refs):
    """Converts an iterable of refs to an iterable of commits they refer to.
    returns: {id : commit}
    """
    commits = dict()
    for ref in refs:
        commit = ref.get_object()
        commits[commit.id] = commit
    return commits.values()


class VisitOnce(object):
    """A yield_fn that remembers things that have been visited."""
    def __init__(self, visited=None, id_fn = lambda x : x):
        self.visited = dict() if visited is None else visited
        self.id = id_fn

    def __call__(self, x):
        id = self.id(x)
        if id in self.visited:
            return False
        else:
            self.visited[id] = x
            return True

# http://codereview.stackexchange.com/questions/78577/depth-first-search-in-python
def dfs(starts, children_fn, yield_fn = lambda x : True):
    """Depth-first search"""
    stack = list(starts)
    while len(stack) > 0:
        vertex = stack.pop()
        if yield_fn(vertex):
            yield vertex
            stack.extend(children_fn(vertex))


# http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
def bfs(starts, children_fn, yield_fn = lambda x : True):
    """Breadth first search"""
    queue = collections.deque(starts)
    while len(queue) > 0:
        vertex = queue.popleft()
        if yield_fn(vertex):
            yield vertex
            queue.extend(children_fn(vertex))


def commit_parents(commit):
    return commit.parents


class DuoGraph(object):
    """Constructs a graph of parents and children, from one with just parents."""

    def children(self, commit):
        return self._children[commit.id]

    @property
    def childless(self):
        return self._childless.values()

    def visit_in_commits(self, x):
        """yield_fn for dfs in __init__"""
        x_id = self.id(x)
        if x_id in self.commits:
            return False
        self.commits[x_id] = x
        return True

    def __init__(self, heads, parents_fn=None, id_fn=lambda x : x):
        self.id = id_fn
        self.parents = parents_fn
        self.commits = dict()        # id --> commit
        self._children = defaultdict(list)    # id --> commit's children
        self.orphans = list()

        for commit in dfs(heads, commit_parents, self.visit_in_commits):
            if len(commit.parents) == 0:
                self.orphans.append(commit)
            else:
                for parent in self.parents(commit):
                    parent_id = self.id(parent)
                    self._children[parent_id].append(commit)

        # Figure out who is childless
        self._childless = dict(self.commits)
        for parent_id in self._children.keys():
            del self._childless[parent_id]

def repo_graph(head_commits):
    """Construct a graph of a repository."""
    return DuoGraph(head_commits,
        parents_fn=lambda commit : commit.parents,
        id_fn = lambda commit : commit.id)

def copy_repo(repo, copy_heads=['master']):
    """Copies the nodes in a repo_graph"""

    branch_prefix = 'origin/'

#    copy_head_commits = [repo.revparse_single(x) for x in copy_heads]
    copy_head_commits = list()
    for x in copy_heads:
        ref = repo.lookup_branch(branch_prefix + x, pygit2.GIT_BRANCH_REMOTE)
        if ref is None:
            ref = repo.lookup_reference('refs/tags/' + x)
        if ref is None:
            raise ValueError('Cannot lookup {}'.format(x))
        copy_head_commits.append(ref.get_object())

    rg = repo_graph(copy_head_commits)
    starts = rg.orphans

    old2new = dict()    # old id --> new commit
    for start in starts:
        old2new[start.id] = start
    visited = dict()
    for old_child in bfs(starts, rg.children, VisitOnce(visited, id_fn=rg.id)):
        repo.checkout(repo.lookup_branch('develop').name)
        if rg.id(old_child) in old2new.keys():
            continue

        old_parents = rg.parents(old_child)
        new_parents = [old2new[rg.id(x)] for x in old_parents]

#        print('old_parents', [x.id for x in old_parents])
#        print('new_parents', [x.id for x in new_parents])

#        print('old: {}'.format(commit2str(old_child)))
        # http://www.pygit2.org/recipes/git-cherry-pick.html
        branch = repo.create_branch('__pgit', new_parents[0], True)
        repo.checkout(branch.name)

        # Cherrypick single commits, re-do merges...
        if len(old_parents) == 1:
            repo.cherrypick(rg.id(old_child))
        else:
            print('MERGE')
            for parent in new_parents:
                repo.merge(rg.id(parent))

        if repo.index.conflicts is not None:
            raise ValueError('Conflicts', rg.id(old_child))

        tree_id = repo.index.write_tree()
        committer = pygit2.Signature('Robot', 'elizabeth.fischer@columbia.edu')
        new_child_id = repo.create_commit(branch.name, old_child.author, old_child.committer,
            'xfer from %s' % old_child.id, tree_id,
            [x.id for x in new_parents])
        new_child = repo.get(new_child_id)
        del branch    # Oudated, prevent accidental use
        repo.state_cleanup()

#        print('new_child', new_child)
        old2new[rg.id(old_child)] = new_child
#        print('    --> %s' % commit2str(old_child))



    # Copy the branches
    for old_branch_str in repo.listall_branches(pygit2.GIT_BRANCH_REMOTE):    # local branches only

        print('old_branch_str', old_branch_str)

        if not old_branch_str.startswith(branch_prefix):
            continue
        old_branch_leaf = old_branch_str[len(branch_prefix):]
        if old_branch_leaf == 'HEAD':
            continue

        # Be paranoid... don't copy again
        if old_branch_leaf.startswith('public/'):
            continue

        print('old_branch_leaf', old_branch_leaf)

        old_branch = repo.lookup_branch(old_branch_str, pygit2.GIT_BRANCH_REMOTE)
        print('old_branch', old_branch)
        old_commit = old_branch.peel(pygit2.Commit)
        old_commit_id = rg.id(old_commit)
        if old_commit_id not in old2new:
            continue

        new_commit = old2new[old_commit_id]
        new_branch_name = 'public/' + old_branch_leaf
        repo.create_branch(new_branch_name, new_commit, True)

    # Copy the tags
#    for tag in repo.listall_references():
#        print(tag)

    tag_prefix = 'refs/tags/'
    for ref_name in  repo.listall_references():
        if not ref_name.startswith(tag_prefix):
            continue
        tag_name = ref_name[len(tag_prefix):]

   regex = re.compile('^refs/tags')
    for tag in filter(lambda r: regex.match(r), ):
        print(tag)


def main():
#    repo = pygit2.Repository('/home2/rpfische/tmp/modelE')
#    for ref in repo.listall_references():
#        print(ref)
#    return


    # Get a list of all refs we're interested in.
#    copy_branch_strs = ['AR5', 'AR5_v2', 'ModelE1-patches', 'develop', 'planet']
    copy_branch_strs = ['_tmp', 'develop', 'docs', 'samplemerge']
    repo = pygit2.Repository('/home2/rpfische/tmp/modele-control')

    copy_repo(repo, copy_branch_strs)
    sys.exit(0)


#    rg = repo_graph([repo.lookup_branch(x) for x in copy_branch_strs])


#    # commit 2d109f602826102123b0c9a9464b2350473541b9
#    # Author: Tom <Thomas.L.Clune@nasa.gov>
#    # Date:   Fri Jan 7 11:09:42 2011 -0500
#    starts = [repo.get('2d109f602826102123b0c9a9464b2350473541b9')]


    print(len(rg.commits))
    common_ancestor = rg.orphans[0]    # Should always be the Ur commit
    src_parent = common_ancestor
    src = rg.children(src_parent)[0]
    out_branch = repo.create_branch('_out', common_ancestor, True)

    index = repo.merge_trees(common_ancestor.tree, out_branch, src, favor='theirs')
    tree_id = index.write_tree(repo)
    repo.create_commit(out_branch.name,
        pygit2.Signature('Archimedes', 'archy@jpl-classics.org'),
        pygit2.Signature('Archimedes', 'archy@jpl-classics.org'),
        'Obtained from other', tree_id, [src_parent.id])


#    print(rg.children(base))
#    print(commit2str(base))
    sys.exit(0)



    base = 'e1f59f3d10a64a44183b8241cce4fc7620c8cd8b'



    refs = []
    refs.append(repo.lookup_reference('refs/heads/master'))
    refs.append(repo.lookup_branch('master'))
    refs.append(repo.lookup_reference('HEAD'))
    print([x.get_object().id for x in refs])
    sys.exit(0)

    ref = repo.lookup_reference("refs/heads/master")
    commit = master_ref.get_object() # or repo[master_ref.target]
    start = commit.parent[0]


#e1f59f3d10a64a44183b8241cce4fc7620c8cd8b
#
#
#    src_parent = 
#
#cherry = repo.revparse_single('9e044d03c')
#basket = repo.lookup_branch('basket')
#
#base      = repo.merge_base(cherry.oid, basket.target)
#base_tree = cherry.parents[0].tree
#
#index = repo.merge_trees(base_tree, basket, cherry)
#tree_id = index.write_tree(repo)
#
#author    = cherry.author
#committer = pygit2.Signature('Archimedes', 'archy@jpl-classics.org')
#
#repo.create_commit(basket.name, author, committer, cherry.message,
#                   tree_id, [basket.target])




    sys.exit(0)

    rg = repo_graph(all_refs_prefix(repo))
    for commit in rg.orphans:
        print(commit2str(commit))
    print(len(rg.commits))

main()







#        # Get initial Commits based on all the heads
#        initial_commits = refs2commits(heads)
#
#        # Trace them back...
#        stack = [x for x in initial_commits.values()]
#        while len(stack) > 0:
#            commit = stack.pop()
#            if commit.id in self.commits:
#                continue
#
#            self.commits[commit.id] = commit
#            if len(commit.parents) == 0:
#                self.orphans.append(commit)
#            else:
#                for parent in commit.parents:
#                    self._children[parent.id] = commit
#                    if parent.id not in self.commits:
#                        stack.append(parent)






#class RepoGraph(object):
#
#    @property
#    def parents(self, commit):
#        return commit.parents
#
#    @property
#    def children(self, commmit):
#        return self.children(commit.id)
#
#    def __init__(self, heads):
#        self.commits = dict()        # id --> commit
#        self._children = defaultdict(list)    # id --> commit's children
#        self.orphans = list()
#
#
#        # Get initial Commits based on all the heads
#        initial_commits = refs2commits(heads)
#
#        # Trace them back...
#        stack = [x for x in initial_commits.values()]
#        while len(stack) > 0:
#            commit = stack.pop()
#            if commit.id in self.commits:
#                continue
#
#            self.commits[commit.id] = commit
#            if len(commit.parents) == 0:
#                self.orphans.append(commit)
#            else:
#                for parent in commit.parents:
#                    self._children[parent.id] = commit
#                    if parent.id not in self.commits:
#                        stack.append(parent)
#
#
#
#
#
##walker = repo.walk(master_commit.id, pygit2.GIT_SORT_TOPOLOGICAL)
#print(sum(1 for _ in walker))
#
#
#
##for commit in repo.walk(commit.id, pygit2.GIT_SORT_TOPOLOGICAL):
##    print(str(commit.oid)[:10], commit.message)
#
##print(commit.oid)
##print(commit.message)
##print(commit.author.name)
##print(commit.author.email)
##print(commit.parents)
#
##refs/heads/master
#
#
#
#all_refs = repo.listall_references()
#
#print('\n'.join(all_refs))
#
#sys.exit(0)
#master_ref = repo.lookup_reference("refs/heads/master")
#master_commit = master_ref.get_object() # or repo[master_ref.target]
#
#commits = dict()        # id --> commit
#children = defaultdict(list)    # id --> commit's children
#
#
#stack = list()
#stack.append(master_commit)
#
#while len(stack) > 0:
#    commit = stack.pop()
#    if commit.id not in commits:
#        commits[commit.id] = commit
#        for parent in commit.parents:
#            children[parent.id] = commit
#            if parent.id not in commits:
#                stack.append(parent)
#
#print(len(commits))
#print(len(children))
#for commit in commits.values():
#    if len(commit.parents) == 0:
#        print(commit.oid)
#        print(commit.message)
#        print(commit.author.name)
#        print(commit.author.email)
#        
#
##children[commit.parents[0].oid] = commit
##print(children)
##commit = commit.parents[0]
#
