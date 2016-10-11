import pygit2
import os
from collections import defaultdict
import sys
import datetime
import collections
import re

branch_location_prefix = 'origin/'



# ----------------------------------------------
class WormDict(dict):
    def __init__(self, *args, eq_fn=lambda x, y : x == y, **kwargs):
        self.eq = eq_fn

#    def __init__(self, inp=None):
#        if isinstance(inp,dict):
#            super(Dict,self).__init__(inp)
#        else:
#            super(Dict,self).__init__()
#            if isinstance(inp, (collections.Mapping, collections.Iterable)): 
#                si = self.__setitem__
#                for k,v in inp:
#                    si(k,v)

    def __setitem__(self, k, v):
        try:
            oldv = self.__getitem__(k)
            if not self.eq(oldv, v):
                raise ValueError("duplicate key '{0}' found".format(k))
        except KeyError:
            super().__setitem__(k,v)
# ----------------------------------------------

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
    def __init__(self, visited=None, id_fn = lambda x : x.id):
        self.visited = dict() if visited is None else visited
        self.id = id_fn

    def __call__(self, x):
        id = self.id(x)
        if id in self.visited:
            return False
        else:
            self.visited[id] = x
            return True

# -------------------------------------------------------
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
# -------------------------------------------------------
sourceRE = re.compile('source: (.*)?.*')
def new_commit_sources(msg):
    """Identify the old commits to which a new commit corresponds.
    Returned in same order they are listed in, in the commit message.

    msg:
        The commit message for the new commit."""

    for line in msg.split('\n'):
        match = sourceRE.match(line)
        if match is None: continue

        yield pygit2.Oid(hex=match.group(1))
# -------------------------------------------------------


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

def get_head_commits(repo, heads, branch_namespace='refs/heads', raise_error=True):
    """Gets the commits corresponding to a user-supplied list of heads
    (eg: ['master', 'develop', etc]

    branch_namespace:
        Prepend this when looking for a branch"""

    refs = []
    for x in heads:
        ref = repo.lookup_reference(branch_namespace + x)
        # ref = repo.lookup_branch(branch_location_prefix + x, pygit2.GIT_BRANCH_REMOTE)
        if ref is None:
            ref = repo.lookup_reference('refs/tags/' + x)
        if ref is None:
            if raise_error:
                raise ValueError('Cannot lookup {}'.format(x))
            else:
                continue
        refs.append(ref)
    commits = [ref.get_object() for ref in refs]
    return commits

def copy_repo(repo, copy_heads=['master']):
    """Copies the nodes in a repo_graph"""

    tag_prefix = 'refs/tags/'
    branch_prefix = 'refs/heads/'

    # -----------------------------------------
    # Look up existing correspondence between old and new commits
    old2new = WormDict(eq_fn = lambda x,y: x.id == y.id)    # old id --> new commit

    new_copy_head_commits = get_head_commits(
        repo, copy_heads, raise_error=False,
        branch_namespace='refs/heads/public/')
    for new_commit in dfs(new_copy_head_commits, commit_parents, VisitOnce()):
        for old_id in new_commit_sources(new_commit.message):
            old2new[old_id] = new_commit
    # -----------------------------------------

    # Get the commits corresponding to the copy_heads:
    copy_head_commits = get_head_commits(repo, copy_heads,
        branch_namespace='refs/remotes/origin/')

    # Set of OIDs that should NOT be collapsed: anything with a tag
    # or a branch on it.
    old_keep_ids = set([commit.id for commit in copy_head_commits])
    for ref_name in repo.listall_references():
        if ref_name.startswith(tag_prefix) or ref_name.startswith(branch_prefix):
            id = repo.lookup_reference(ref_name).get_object().id
            old_keep_ids.add(id)



    rg = repo_graph(copy_head_commits)
    starts = rg.orphans

    # Translation from old commits to new commits
    for start in starts:
        old2new[start.id] = start    # Share the orphans
    visited = dict()
    for old_child in bfs(starts, rg.children, VisitOnce(visited, id_fn=rg.id)):
        repo.checkout(repo.lookup_branch('develop').name)

        # Don't transfer nodes already transferred.
        if rg.id(old_child) in old2new.keys():
            continue

        old_parents = rg.parents(old_child)
        new_parents = [old2new[rg.id(x)] for x in old_parents]

        # Use the same tree as the old commit...
        tree = old_child.tree

        # Remove everything but the lib folder...
        tb = repo.TreeBuilder(tree)
        for entry in tree:
            if entry.name != 'lib':
                tb.remove(entry.name)
        new_tree_id = tb.write()

#        branch = repo.create_branch('__pgit', new_parents[0], True)

        committer = pygit2.Signature('Robot', 'elizabeth.fischer@columbia.edu')
        new_child_id = repo.create_commit(branch.name, old_child.author, old_child.committer,
            'source: %s' % old_child.id, new_tree_id,
            [x.id for x in new_parents])
        new_child = repo.get(new_child_id)

        print('Copied node: {} -> {}'.format(old_child.id, new_child.id))
        old2new[rg.id(old_child)] = new_child



    # Copy the branches
    for old_branch_str in repo.listall_branches(pygit2.GIT_BRANCH_REMOTE):    # local branches only

        if not old_branch_str.startswith(branch_location_prefix):
            continue

        old_branch_leaf = old_branch_str[len(branch_location_prefix):]
        if old_branch_leaf == 'HEAD':
            continue

        # Be paranoid... don't copy again
        if old_branch_leaf.startswith('public/'):
            continue

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
    # http://ben.straub.cc/2013/06/03/refs-tags-and-branching/
#    for tag in repo.listall_references():
#        print(tag)

    for ref_name in  repo.listall_references():
        print(ref_name)
        if not ref_name.startswith(tag_prefix):
            continue
        if ref_name.startswith(tag_prefix + 'public/'):
            continue

        old_tag_name = ref_name[len(tag_prefix):]
        new_tag_name = 'public/' + old_tag_name


        print('old_tag_name', old_tag_name)
        print('ref_name', ref_name)
        old_ref = repo.lookup_reference(ref_name)#.get_object()

        try:
            # Annotated tag
            old_tag = old_ref.peel(pygit2.Tag)
            old_target_id = old_tag.target
        except ValueError:
            # Non-annotated tag
            old_target_id = old_ref.target
            old_tag = None

        if old_target_id not in old2new:
            continue

        # Lookup new_tag_name
        try:
            new_ref = repo.lookup_reference(tag_prefix + new_tag_name)
        except KeyError:
            new_ref = None

        new_target_id = old2new[old_target_id].id
        if new_ref is not None and new_ref.target != new_target_id:
            print('Deleteing tag', new_ref.target)
            new_ref.delete()
            new_ref = None
        if new_ref is None:
            print('Creating tag...')

            if old_tag is not None:
                new_tagger = old_tag.tagger
                new_message = old_tag.message
                new_message = ''    # Clear it out
            else:
                new_tagger = pygit2.Signature('Robot', 'elizabeth.fischer@columbia.edu')
                new_message = ''    # There never was a message

            # Create annotated tag
            new_ref = repo.create_tag(
                new_tag_name,
                new_target_id, pygit2.GIT_OBJ_COMMIT,
                new_tagger, new_message)

            print(new_tag_name)


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

main()
